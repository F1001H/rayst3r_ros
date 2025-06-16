from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import sys
import open3d as o3d
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAYST3R_PATH = os.path.join(SCRIPT_DIR, 'rayst3r')
sys.path.insert(0, RAYST3R_PATH)
sys.path.insert(0, SCRIPT_DIR)

from eval_wrapper.sample_poses import pointmap_to_poses
from utils.fusion import fuse_batch
from models.rayquery import *
from models.losses import *
import argparse
from utils import misc
import torch.distributed as dist
from utils.collate import collate
from engine import eval_model
from utils.viz import just_load_viz
from utils.geometry import compute_pointmap_torch
from eval_wrapper.eval_utils import npy2ply, filter_all_masks
from huggingface_hub import hf_hub_download
import copy

class Rayst3rModelWrapper(torch.nn.Module):
    def __init__(self,checkpoint_path,distributed=False,device="cuda",dtype=torch.float32,**kwargs):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_string = checkpoint['args'].model
        
        self.model = eval(model_string).to(device)
        if distributed:
            rank, world_size, local_rank = misc.setup_distributed()
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank],find_unused_parameters=True)
        
        self.dtype = dtype
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def forward(self,x,dino_model=None):
        pred, gt, loss, scale = eval_model(self.model,x,mode='viz',dino_model=dino_model,return_scale=True)
        return pred, gt, loss, scale

class PostProcessWrapper(torch.nn.Module):
    def __init__(self,pred_mask_threshold = 0.5, mode='novel_views',
    debug=False,conf_dist_mode='isotonic',set_conf=None,percentile=20,
    no_input_mask=False,no_pred_mask=False):
        super().__init__()
        self.pred_mask_threshold = pred_mask_threshold
        self.mode = mode
        self.debug = debug
        self.conf_dist_mode = conf_dist_mode
        self.set_conf = set_conf
        self.percentile = percentile
        self.no_input_mask = no_input_mask
        self.no_pred_mask = no_pred_mask

    def transform_pointmap(self,pointmap_cam,c2w):
        pointmap_cam_h = torch.cat([pointmap_cam,torch.ones(pointmap_cam.shape[:-1]+(1,)).to(pointmap_cam.device)],dim=-1)
        pointmap_world_h = pointmap_cam_h @ c2w.T
        pointmap_world = pointmap_world_h[...,:3]/pointmap_world_h[...,3:4]
        return pointmap_world

    def reject_conf_points(self,conf_pts):
        if self.set_conf is None:
            raise ValueError("set_conf must be set")
        
        conf_mask = conf_pts > self.set_conf
        return conf_mask
    
    
    def project_input_mask(self,pred_dict,batch):
        input_mask = batch['input_cams']['original_valid_masks'][0][0]
        input_c2w = batch['input_cams']['c2ws'][0][0]
        input_w2c = torch.linalg.inv(input_c2w)
        input_K = batch['input_cams']['Ks'][0][0]
        H, W = input_mask.shape
        pointmaps_input_cam = torch.stack([self.transform_pointmap(pmap,input_w2c@c2w) for pmap,c2w in zip(pred_dict['pointmaps'][0],batch['new_cams']['c2ws'][0])]) 
        img_coords = pointmaps_input_cam @ input_K.T
        img_coords = (img_coords[...,:2]/img_coords[...,2:3]).int()

        n_views, H, W = img_coords.shape[:3]
        device = input_mask.device
        if self.no_input_mask:
            combined_mask = torch.ones((n_views, H, W), device=device)
        else:
            combined_mask = torch.zeros((n_views, H, W), device=device)

            xs = img_coords[..., 0].view(n_views, -1)
            ys = img_coords[..., 1].view(n_views, -1)

            i_coords = torch.arange(H, device=device).view(-1, 1).expand(H, W).reshape(-1)
            j_coords = torch.arange(W, device=device).view(1, -1).expand(H, W).reshape(-1)
            mask_coords = torch.stack((i_coords, j_coords), dim=-1)

            valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

            xs_clipped = torch.clamp(xs, 0, W-1)
            ys_clipped = torch.clamp(ys, 0, H-1)

            flat_input_mask = input_mask[ys_clipped, xs_clipped]
            input_mask_mask = flat_input_mask & valid

            depth_points = pointmaps_input_cam[..., -1].view(n_views, -1)
            input_depths = batch['input_cams']['depths'][0][0][ys_clipped, xs_clipped]

            depth_mask = (depth_points > input_depths) & input_mask_mask

            final_i = mask_coords[:, 0].unsqueeze(0).expand(n_views, -1)[depth_mask]
            final_j = mask_coords[:, 1].unsqueeze(0).expand(n_views, -1)[depth_mask]
            final_view_idx = torch.arange(n_views, device=device).view(-1, 1).expand(-1, H*W)[depth_mask]

            combined_mask[final_view_idx, final_i, final_j] = 1 
        return combined_mask.unsqueeze(0).bool()

    def forward(self,pred_dict,batch):
        if self.mode == 'novel_views':
            project_masks = self.project_input_mask(pred_dict,batch)
            pred_mask_raw = torch.sigmoid(pred_dict['classifier'])
            if self.no_pred_mask:
                pred_masks = torch.ones_like(project_masks).bool()
            else:
                pred_masks = (pred_mask_raw > self.pred_mask_threshold).bool()
            
            conf_masks = self.reject_conf_points(pred_dict['conf_pointmaps'])
            combined_mask = project_masks & pred_masks & conf_masks
            batch['new_cams']['valid_masks'] = combined_mask 

        elif self.mode == 'input_view':
            conf_masks = self.reject_conf_points(pred_dict['conf_pointmaps'])
            if self.no_pred_mask:
                pred_masks = torch.ones_like(conf_masks).bool()
            else:
                pred_mask_raw = torch.sigmoid(pred_dict['classifier'])
                pred_masks = (pred_mask_raw > self.pred_mask_threshold).bool()
            combined_mask = conf_masks & batch['new_cams']['valid_masks'] & pred_masks
            batch['new_cams']['valid_masks'] = combined_mask

        return pred_dict, batch


class GenericLoaderSmall(Dataset):
    def __init__(self,
                 rgb_image_np,
                 depth_image_np_metric,
                 mask_image_np,
                 intrinsics_np_or_tensor,
                 c2w_original_np_or_tensor=None,
                 data_dir=None,
                 mode="single_scene",
                 dtype=torch.float32,
                 n_pred_views=3,
                 pred_input_only=False,
                 min_depth=0.0,
                 pointmap_for_bb=None,
                 run_octmae=False,
                 ):
        self.rgb_data = rgb_image_np
        self.depth_metric_data = depth_image_np_metric
        self.mask_data = mask_image_np
        
        if isinstance(intrinsics_np_or_tensor, np.ndarray):
            self.K_data = torch.from_numpy(intrinsics_np_or_tensor.astype(np.float32))
        elif isinstance(intrinsics_np_or_tensor, torch.Tensor):
            self.K_data = intrinsics_np_or_tensor.float()
        else:
            raise TypeError("intrinsics_np_or_tensor must be a NumPy array or PyTorch Tensor.")

        if c2w_original_np_or_tensor is None:
            self.c2w_original_data = torch.eye(4, dtype=dtype)
        elif isinstance(c2w_original_np_or_tensor, np.ndarray):
            self.c2w_original_data = torch.from_numpy(c2w_original_np_or_tensor.astype(np.float32))
        elif isinstance(c2w_original_np_or_tensor, torch.Tensor):
            self.c2w_original_data = c2w_original_np_or_tensor.to(dtype)
        else:
            raise TypeError("c2w_original_np_or_tensor must be a NumPy array, PyTorch Tensor, or None.")

        self.data_dir_param = data_dir
        self.mode = mode
        self.dtype = dtype
        self.rng = np.random.RandomState(seed=42)
        self.n_pred_views = n_pred_views
        
        self.min_depth_scaled = self.depth_metric_to_uint16(torch.tensor(min_depth, dtype=torch.float32))

        self.inputs = [0]

        self.pred_input_only = pred_input_only
        if self.pred_input_only:
            self.n_pred_views = 1
        
        self.desired_resolution = (480, 640)
        try:
            self.resize_transform_rgb = transforms.Resize(self.desired_resolution, antialias=True)
        except TypeError:
            self.resize_transform_rgb = transforms.Resize(self.desired_resolution)
        self.resize_transform_depth = transforms.Resize(self.desired_resolution, interpolation=transforms.InterpolationMode.NEAREST)
        
        self.pointmap_for_bb = pointmap_for_bb
        self.run_octmae = run_octmae

    def transform_pointmap(self, pointmap_cam, c2w):
        pointmap_cam_h = torch.cat([pointmap_cam, torch.ones(pointmap_cam.shape[:-1]+(1,)).to(pointmap_cam.device)],dim=-1)
        pointmap_world_h = pointmap_cam_h @ c2w.T
        pointmap_world = pointmap_world_h[...,:3]/pointmap_world_h[...,3:4]
        return pointmap_world

    def __len__(self):
        return len(self.inputs)

    def look_at(self, cam_pos, center=(0,0,0), up=(0,0,1)):
        z = np.array(center) - np.array(cam_pos)
        z_norm = np.linalg.norm(z, axis=-1, keepdims=True)
        if np.any(z_norm == 0):
            if np.all(cam_pos == center):
                 return np.array([[1,0,0,cam_pos[0]],[0,-1,0,cam_pos[1]],[0,0,-1,cam_pos[2]],[0,0,0,1]], dtype=np.float32)

        z = z / z_norm
        y = -np.array(up, dtype=np.float32)
        y = y - np.sum(y * z, axis=-1, keepdims=True) * z
        y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
        if np.any(y_norm == 0):
            if np.allclose(np.abs(z), [0,0,1]): y = np.array([0,1,0], dtype=np.float32)
            else: y = np.cross(z, np.array([1,0,0], dtype=np.float32)); y = y/np.linalg.norm(y)

        else:
            y = y / y_norm

        x = np.cross(y, z, axis=-1)
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
        if np.any(x_norm == 0):
            pass
        else:
            x = x/x_norm


        cam2w = np.eye(4, dtype=np.float32)
        cam2w[:3,0] = x
        cam2w[:3,1] = y
        cam2w[:3,2] = z
        cam2w[:3,3] = cam_pos
        return cam2w.astype(np.float32)


    def find_new_views(self, n_views, geometric_median=(0,0,0), r_min=0.4, r_max=0.9):
        rad = self.rng.uniform(r_min, r_max, size=n_views)
        azi = self.rng.uniform(0, 2*np.pi, size=n_views)
        ele = self.rng.uniform(-np.pi/2 + 0.1, np.pi/2 - 0.1, size=n_views)
        
        x_rel = rad * np.cos(ele) * np.cos(azi)
        y_rel = rad * np.cos(ele) * np.sin(azi)
        z_rel = rad * np.sin(ele)
        
        cam_centers = np.c_[x_rel, y_rel, z_rel] + np.array(geometric_median)
        
        c2ws = [self.look_at(cam_pos=cam_center, center=geometric_median) for cam_center in cam_centers]
        return c2ws

    def depth_uint16_to_metric(self, depth_scaled_tensor):
        return depth_scaled_tensor / (torch.iinfo(torch.uint16).max / 10.0)

    def depth_metric_to_uint16(self, depth_metric_tensor):
        return depth_metric_tensor * (torch.iinfo(torch.uint16).max / 10.0)

    def resize(self, depth_tensor, img_tensor, mask_tensor, K_tensor_orig):
        K_tensor = K_tensor_orig.clone()

        s_x = float(self.desired_resolution[1]) / img_tensor.shape[1]
        s_y = float(self.desired_resolution[0]) / img_tensor.shape[0]

        depth_r = self.resize_transform_depth(depth_tensor.unsqueeze(0)).squeeze(0)
        
        img_chw_float = (img_tensor.permute(2,0,1).float() / 255.0)
        img_r_chw_float = self.resize_transform_rgb(img_chw_float)
        img_r_hwc_byte = (img_r_chw_float * 255.0).byte().permute(1,2,0)

        mask_r = self.resize_transform_depth(mask_tensor.unsqueeze(0).float()).squeeze(0).bool()
        
        K_tensor[0,0] *= s_x; K_tensor[0,2] *= s_x
        K_tensor[1,1] *= s_y; K_tensor[1,2] *= s_y
    
        return depth_r, img_r_hwc_byte, mask_r, K_tensor
    
    def __getitem__(self, idx):

        data = dict(new_cams={}, input_cams={})

        data['input_cams']['c2ws_original'] = [self.c2w_original_data.to(self.dtype)]
        
        data['input_cams']['c2ws'] = [torch.eye(4).to(self.dtype)]
        
        initial_K = self.K_data.clone().to(self.dtype)
        
        initial_depth_metric = torch.from_numpy(self.depth_metric_data.astype(np.float32))
        initial_depth_scaled = self.depth_metric_to_uint16(initial_depth_metric)
        
        initial_mask = torch.from_numpy(self.mask_data.astype(np.bool_))
        initial_img = torch.from_numpy(self.rgb_data.astype(np.uint8))
        
        if initial_depth_scaled.shape[0] != self.desired_resolution[0] or \
           initial_depth_scaled.shape[1] != self.desired_resolution[1]:
            
            resized_depth_scaled, resized_img, resized_mask, resized_K = \
                self.resize(initial_depth_scaled, initial_img, initial_mask, initial_K)
        else:
            resized_depth_scaled = initial_depth_scaled
            resized_img = initial_img
            resized_mask = initial_mask
            resized_K = initial_K

        data['input_cams']['Ks'] = [resized_K]
        data['input_cams']['depths'] = [resized_depth_scaled]
        data['input_cams']['valid_masks'] = [resized_mask]
        data['input_cams']['imgs'] = [resized_img]

        data['input_cams']['original_valid_masks'] = [resized_mask.clone()]
        
        data['input_cams']['valid_masks'][0] = data['input_cams']['valid_masks'][0] & \
                                               (data['input_cams']['depths'][0] > self.min_depth_scaled)

        if self.pred_input_only:
            c2ws_for_new_cams_np = [data['input_cams']['c2ws'][0].cpu().numpy()]
            actual_n_pred_views = 1
        else:
            input_mask_for_pointmap = data['input_cams']['valid_masks'][0]
            
            if self.pointmap_for_bb is not None:
                pointmap_input_world = self.pointmap_for_bb
            else:
                depth_for_pointmap_metric = self.depth_uint16_to_metric(data['input_cams']['depths'][0])
                c2w_for_pointmap = data['input_cams']['c2ws'][0]
                K_for_pointmap = data['input_cams']['Ks'][0]
                
                pointmap_cam_coords = compute_pointmap_torch(
                    depth_for_pointmap_metric,
                    c2w_for_pointmap,
                    K_for_pointmap,
                    device='cpu'
                )
                pointmap_input_cam = pointmap_cam_coords[input_mask_for_pointmap]
                pointmap_input_world = pointmap_input_cam
            
            if pointmap_input_world.shape[0] == 0:
                print(
                    "Warning: Point map for novel pose sampling is empty. "
                    "This means no valid points were found or provided. "
                    "Generating default fallback poses around origin."
                )
                c2ws_for_new_cams_np = self.find_new_views(self.n_pred_views, geometric_median=(0,0,0.5))
                if not c2ws_for_new_cams_np:
                     c2ws_for_new_cams_np = [np.eye(4, dtype=np.float32)]
            else:
                c2ws_for_new_cams_np = pointmap_to_poses(
                    pointmap_input_world,
                    self.n_pred_views,
                    inner_radius=1.1, outer_radius=2.5,
                    device='cpu',
                    run_octmae=self.run_octmae
                )
            
            valid_poses_generated = False
            if isinstance(c2ws_for_new_cams_np, list) and len(c2ws_for_new_cams_np) > 0:
                valid_poses_generated = True
            elif isinstance(c2ws_for_new_cams_np, np.ndarray):
                if c2ws_for_new_cams_np.ndim == 3 and c2ws_for_new_cams_np.shape[0] > 0:
                    c2ws_for_new_cams_np = [arr for arr in c2ws_for_new_cams_np]
                    valid_poses_generated = True
                elif c2ws_for_new_cams_np.ndim == 2 and c2ws_for_new_cams_np.shape == (4,4):
                    c2ws_for_new_cams_np = [c2ws_for_new_cams_np]
                    valid_poses_generated = True

            if not valid_poses_generated:
                print("Warning: pointmap_to_poses (or fallback) returned no valid poses. Using single identity pose as fallback.")
                c2ws_for_new_cams_np = [np.eye(4, dtype=np.float32)]

            actual_n_pred_views = len(c2ws_for_new_cams_np)

        data['new_cams']['c2ws'] = [torch.from_numpy(c2w).to(self.dtype) for c2w in c2ws_for_new_cams_np]
        data['new_cams']['Ks'] = [data['input_cams']['Ks'][0].clone() for _ in range(actual_n_pred_views)]
        
        dummy_depth_shape = data['input_cams']['depths'][0].shape
        dummy_mask_shape = data['input_cams']['valid_masks'][0].shape

        data['new_cams']['depths'] = [torch.zeros(dummy_depth_shape, dtype=data['input_cams']['depths'][0].dtype) for _ in range(actual_n_pred_views)]
        
        if self.pred_input_only:
            data['new_cams']['valid_masks'] = [data['input_cams']['original_valid_masks'][0].clone()]
        else:
            data['new_cams']['valid_masks'] = [torch.ones(dummy_mask_shape, dtype=torch.bool) for _ in range(actual_n_pred_views)]
        
        return data

def dict_to_float(d):
    return {k: v.float() for k, v in d.items()}

def merge_dicts(d1,d2):
    for k,v in d1.items():
        d1[k] = torch.cat([d1[k],d2[k]],dim=1)
    return d1

def compute_all_points(pred_dict,batch):
    n_views = pred_dict['depths'].shape[1]
    all_points = None 
    for i in range(n_views):
        mask = batch['new_cams']['valid_masks'][0,i]
        pointmap = compute_pointmap_torch(pred_dict['depths'][0,i],batch['new_cams']['c2ws'][0,i],batch['new_cams']['Ks'][0,i])
        masked_points = pointmap[mask]
        if all_points is None:
            all_points = masked_points
        else:
            all_points = torch.cat([all_points,masked_points],dim=0)
    return all_points

def infer_scene_points(model, rgb_image_np, depth_image_np_metric, mask_image_np, 
        intrinsics_np, c2w_original_np=np.eye,run_octmae=False,set_conf=5,
               no_input_mask=False,no_pred_mask=False,no_filter_input_view=False,n_pred_views=5,
               do_filter_all_masks=False, dino_model=None):
    
    if dino_model is None:
        dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14_reg")
        dino_model.eval()
        dino_model.to("cuda")

    dataloader_input_view = GenericLoaderSmall(
        rgb_image_np=rgb_image_np,
        depth_image_np_metric=depth_image_np_metric,
        mask_image_np=mask_image_np,
        intrinsics_np_or_tensor=intrinsics_np,
        c2w_original_np_or_tensor=c2w_original_np,
        pred_input_only=True,
    )
    input_view_loader = DataLoader(dataloader_input_view, batch_size=1, shuffle=True, collate_fn=collate)
    input_view_batch = next(iter(input_view_loader))

    postprocessor_input_view = PostProcessWrapper(mode='input_view',set_conf=set_conf,
                                                  no_input_mask=no_input_mask,no_pred_mask=no_pred_mask)
    postprocessor_pred_views = PostProcessWrapper(mode='novel_views',debug=False,set_conf=set_conf,
                                                  no_input_mask=no_input_mask,no_pred_mask=no_pred_mask)
    fused_meshes = None
    with torch.no_grad():
        pred_input_view, gt_input_view, _, scale_factor = model(input_view_batch,dino_model)
        if no_filter_input_view:
            pred_input_view['pointmaps'] = input_view_batch['input_cams']['pointmaps']
            pred_input_view['depths'] = input_view_batch['input_cams']['depths']
        else: 
            pred_input_view, input_view_batch = postprocessor_input_view(pred_input_view,input_view_batch)

        input_points = pred_input_view['pointmaps'][0][0][input_view_batch['new_cams']['valid_masks'][0][0]] * (1.0/scale_factor)
        if input_points.shape[0] == 0:
            input_points = None
        
        dataloader_pred_views = GenericLoaderSmall(
        rgb_image_np=rgb_image_np,
        depth_image_np_metric=depth_image_np_metric,
        mask_image_np=mask_image_np,
        intrinsics_np_or_tensor=intrinsics_np,
        c2w_original_np_or_tensor=c2w_original_np,
        pred_input_only=False,
    )
        pred_views_loader = DataLoader(dataloader_pred_views, batch_size=1, shuffle=True, collate_fn=collate)
        pred_views_batch = next(iter(pred_views_loader))

        pred_new_views, gt_new_views, _, scale_factor = model(pred_views_batch,dino_model)
        pred_new_views, pred_views_batch = postprocessor_pred_views(pred_new_views,pred_views_batch)
    
    pred = merge_dicts(dict_to_float(pred_input_view),dict_to_float(pred_new_views))
    gt = merge_dicts(dict_to_float(gt_input_view),dict_to_float(gt_new_views))

    batch = copy.deepcopy(input_view_batch)
    batch['new_cams'] = merge_dicts(input_view_batch['new_cams'],pred_views_batch['new_cams'])
    gt['pointmaps'] = None
    
    if do_filter_all_masks:
        batch = filter_all_masks(pred,input_view_batch,max_outlier_views=1)

    all_points = compute_all_points(pred,batch)
    all_points = all_points*(1.0/scale_factor)
    
    all_points_h = torch.cat([all_points,torch.ones(all_points.shape[:-1]+(1,)).to(all_points.device)],dim=-1)
    all_points_original = all_points_h @ batch['input_cams']['c2ws_original'][0][0].T
    all_points = all_points_original[...,:3]
    return all_points

def run_inference(rgb_image_np,
                  depth_image_np,
                  mask_image_np,
                  intrinsics_np,
                  c2w_np=None,
                  n_pred_views=5,
                  set_conf=5.0,
                  no_input_mask=False,
                  no_pred_mask=False,
                  no_filter_input_view=False,
                  run_octmae=False,
                  output_filename="inference_points.ply"):


    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    rayst3r_checkpoint_path = hf_hub_download("bartduis/rayst3r", "rayst3r.pth")
    model = Rayst3rModelWrapper(rayst3r_checkpoint_path, distributed=False, device=device_str)
    
    dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14_reg", verbose=False)
    dino_model.eval()
    dino_model.to(device_str)

    if not c2w_np is None:
        c2w_original_np = c2w_np
    else:
        c2w_original_np = np.eye(4)

    all_points_tensor = infer_scene_points(
        model=model,
        dino_model=dino_model,
        rgb_image_np=rgb_image_np,
        depth_image_np_metric=depth_image_np,
        mask_image_np=mask_image_np,
        intrinsics_np=intrinsics_np,
        c2w_original_np=c2w_original_np,
        run_octmae=run_octmae,
        set_conf=set_conf,    
        no_input_mask=no_input_mask, 
        no_pred_mask=no_pred_mask,   
        no_filter_input_view=no_filter_input_view, 
        n_pred_views=n_pred_views, 
    )

    if all_points_tensor is None or all_points_tensor.shape[0] == 0:
        if device_str == "cuda": torch.cuda.empty_cache()
        return None

    all_points_np = all_points_tensor.cpu().numpy()
    all_points_save_path = os.path.join(output_filename)
    o3d_pc = npy2ply(all_points_np, colors=None, normals=None)
    o3d.io.write_point_cloud(all_points_save_path, o3d_pc)

    if device_str == "cuda":
        torch.cuda.empty_cache()
    return o3d_pc