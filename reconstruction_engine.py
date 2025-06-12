from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
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
        input_mask = batch['input_cams']['original_valid_masks'][0][0] # shape H x W
        input_c2w = batch['input_cams']['c2ws'][0][0]
        input_w2c = torch.linalg.inv(input_c2w)
        input_K = batch['input_cams']['Ks'][0][0]
        H, W = input_mask.shape
        pointmaps_input_cam = torch.stack([self.transform_pointmap(pmap,input_w2c@c2w) for pmap,c2w in zip(pred_dict['pointmaps'][0],batch['new_cams']['c2ws'][0])]) # bp: Assuming batch size is 1!!
        img_coords = pointmaps_input_cam @ input_K.T
        img_coords = (img_coords[...,:2]/img_coords[...,2:3]).int()

        n_views, H, W = img_coords.shape[:3]
        device = input_mask.device
        if self.no_input_mask:
            combined_mask = torch.ones((n_views, H, W), device=device)
        else:
            combined_mask = torch.zeros((n_views, H, W), device=device)

            # Flatten spatial dims
            xs = img_coords[..., 0].view(n_views, -1)  # [V, H*W]
            ys = img_coords[..., 1].view(n_views, -1)  # [V, H*W]

            # Create base pixel coords (i, j)
            i_coords = torch.arange(H, device=device).view(-1, 1).expand(H, W).reshape(-1)  # [H*W]
            j_coords = torch.arange(W, device=device).view(1, -1).expand(H, W).reshape(-1)  # [H*W]
            mask_coords = torch.stack((i_coords, j_coords), dim=-1)  # [H*W, 2], shared across views

            # Mask for valid projections
            valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)  # [V, H*W]

            # Clip out-of-bounds coords for indexing (only valid will be used anyway)
            xs_clipped = torch.clamp(xs, 0, W-1)
            ys_clipped = torch.clamp(ys, 0, H-1)

            # input_mask lookup per view
            flat_input_mask = input_mask[ys_clipped, xs_clipped]  # [V, H*W]
            input_mask_mask = flat_input_mask & valid  # apply valid range mask

            # Apply mask to coords and depths
            depth_points = pointmaps_input_cam[..., -1].view(n_views, -1)  # [V, H*W]
            input_depths = batch['input_cams']['depths'][0][0][ys_clipped, xs_clipped]  # [V, H*W]

            depth_mask = (depth_points > input_depths) & input_mask_mask  # final mask [V, H*W]
            #depth_mask = input_mask_mask  # final mask [V, H*W]

            # Get final (i,j) coords to write
            final_i = mask_coords[:, 0].unsqueeze(0).expand(n_views, -1)[depth_mask]  # [N_mask]
            final_j = mask_coords[:, 1].unsqueeze(0).expand(n_views, -1)[depth_mask]  # [N_mask]
            final_view_idx = torch.arange(n_views, device=device).view(-1, 1).expand(-1, H*W)[depth_mask]  # [N_mask]

            # Scatter final mask
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
            batch['new_cams']['valid_masks'] = combined_mask # this is for visualization

        return pred_dict, batch

class GenericLoaderSmall(torch.utils.data.Dataset):
    def __init__(self,data_dir,mode="single_scene",dtype=torch.float32,n_pred_views=3,pred_input_only=False,min_depth=0.0,
    pointmap_for_bb=None,run_octmae=False,false_positive=None,false_negative=None):
        self.data_dir = data_dir
        self.mode = mode
        self.dtype = dtype
        self.rng = np.random.RandomState(seed=42)
        self.n_pred_views = n_pred_views
        self.min_depth = self.depth_metric_to_uint16(min_depth)
        if self.mode == "single_scene":
            self.inputs = [data_dir]
        self.pred_input_only = pred_input_only
        if self.pred_input_only:
            self.n_pred_views = 1
        self.desired_resolution = (480,640)
        self.resize_transform_rgb = transforms.Resize(self.desired_resolution)
        self.resize_transform_depth = transforms.Resize(self.desired_resolution,interpolation=transforms.InterpolationMode.NEAREST)
        self.pointmap_for_bb = pointmap_for_bb
        self.run_octmae = run_octmae
    
    def transform_pointmap(self,pointmap_cam,c2w):
        pointmap_cam_h = torch.cat([pointmap_cam,torch.ones(pointmap_cam.shape[:-1]+(1,)).to(pointmap_cam.device)],dim=-1)
        pointmap_world_h = pointmap_cam_h @ c2w.T
        pointmap_world = pointmap_world_h[...,:3]/pointmap_world_h[...,3:4]
        return pointmap_world

    def __len__(self):
        return len(self.inputs)
    
    def look_at(self,cam_pos, center=(0,0,0), up=(0,0,1)):
        z = center - cam_pos
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        y = -np.float32(up)
        y = y - np.sum(y * z, axis=-1, keepdims=True) * z
        y /= np.linalg.norm(y, axis=-1, keepdims=True)
        x = np.cross(y, z, axis=-1)

        cam2w = np.r_[np.c_[x,y,z,cam_pos],[[0,0,0,1]]]
        return cam2w.astype(np.float32)

    def find_new_views(self,n_views,geometric_median = (0,0,0),r_min=0.4,r_max=0.9):
        rad = self.rng.uniform(r_min,r_max, size=n_views)
        azi = self.rng.uniform(0, 2*np.pi, size=n_views)
        ele = self.rng.uniform(-np.pi, np.pi, size=n_views)
        cam_centers = np.c_[np.cos(azi), np.sin(azi)] 
        cam_centers = rad[:,None] * np.c_[np.cos(ele)[:,None]*cam_centers, np.sin(ele)] + geometric_median
        
        c2ws = [self.look_at(cam_pos=cam_center,center=geometric_median) for cam_center in cam_centers]
        return c2ws

    def depth_uint16_to_metric(self,depth):
        return depth / torch.iinfo(torch.uint16).max * 10.0 # threshold is in m, convert to uint16 value

    def depth_metric_to_uint16(self,depth):
        return depth * torch.iinfo(torch.uint16).max / 10.0 # threshold is in m, convert to uint16 value

    def resize(self,depth,img,mask,K):
        s_x = self.desired_resolution[1] / img.shape[1]
        s_y = self.desired_resolution[0] / img.shape[0]
        depth = self.resize_transform_depth(depth.unsqueeze(0)).squeeze(0)
        img = self.resize_transform_rgb(img.permute(-1,0,1)).permute(1,2,0)
        mask = self.resize_transform_depth(mask.unsqueeze(0)).squeeze(0)
        K[0] *= s_x
        K[1] *= s_y
        return depth, img, mask, K
    

    def __getitem__(self,idx):
        scene_dir = self.inputs[idx]
        
        data = dict(new_cams={},input_cams={})

        c2w_path = os.path.join(scene_dir,'cam2world.pt')
        if os.path.exists(c2w_path):
            data['input_cams']['c2ws_original'] = [torch.load(c2w_path,map_location='cpu',weights_only=True).to(self.dtype)]
        else:
            data['input_cams']['c2ws_original'] = [torch.eye(4).to(self.dtype)]
        
        data['input_cams']['c2ws'] = [torch.eye(4).to(self.dtype)]
        data['input_cams']['Ks'] = [torch.load(os.path.join(scene_dir,'intrinsics.pt'),map_location='cpu',weights_only=True).to(self.dtype)]
        data['input_cams']['depths'] = [torch.from_numpy(np.array(Image.open(os.path.join(scene_dir,'depth.png'))).astype(np.float32))]
        data['input_cams']['valid_masks'] = [torch.from_numpy(np.array(Image.open(os.path.join(scene_dir,'mask.png')))).bool()]
        data['input_cams']['imgs'] = [torch.from_numpy(np.array(Image.open(os.path.join(scene_dir,'rgb.png'))))]

        if data['input_cams']['depths'][0].shape != self.desired_resolution:
            data['input_cams']['depths'][0], data['input_cams']['imgs'][0], data['input_cams']['valid_masks'][0], data['input_cams']['Ks'][0] = \
            self.resize(data['input_cams']['depths'][0], data['input_cams']['imgs'][0], data['input_cams']['valid_masks'][0], data['input_cams']['Ks'][0])
        
        data['input_cams']['original_valid_masks'] = [data['input_cams']['valid_masks'][0].clone()]
        data['input_cams']['valid_masks'][0] = data['input_cams']['valid_masks'][0] & \
            (data['input_cams']['depths'][0] > self.min_depth)

        if self.pred_input_only:
            c2ws = [data['input_cams']['c2ws'][0].cpu().numpy()]
        else:
            input_mask = data['input_cams']['valid_masks'][0]
            if self.pointmap_for_bb is not None:
                pointmap_input = self.pointmap_for_bb
            else:
                pointmap_input = compute_pointmap_torch(self.depth_uint16_to_metric(data['input_cams']['depths'][0]),data['input_cams']['c2ws'][0],data['input_cams']['Ks'][0],device='cpu')[input_mask]
            if pointmap_input.shape[0] == 0:
                    raise ValueError(
                        "Point map for pose sampling is empty. This means no valid points were found "
                        "in the intersection of 'mask.png' and the depth map (with min_depth > 0.1m). "
                        "Please check your input data."
                    )
            c2ws = pointmap_to_poses(pointmap_input, self.n_pred_views, inner_radius=1.1, outer_radius=2.5, device='cpu',run_octmae=self.run_octmae)
            self.n_pred_views = len(c2ws)
        
        data['new_cams'] = {}
        data['new_cams']['c2ws'] = [torch.from_numpy(c2w).to(self.dtype) for c2w in c2ws]
        data['new_cams']['depths'] = [torch.zeros_like(data['input_cams']['depths'][0]) for _ in range(self.n_pred_views)]
        data['new_cams']['Ks'] = [data['input_cams']['Ks'][0] for _ in range(self.n_pred_views)]
        if self.pred_input_only:
            data['new_cams']['valid_masks'] = data['input_cams']['original_valid_masks']
        else:
            data['new_cams']['valid_masks'] = [torch.ones_like(data['input_cams']['valid_masks'][0]) for _ in range(self.n_pred_views)]
        
        return data

def dict_to_float(d):
    return {k: v.float() for k, v in d.items()}

def merge_dicts(d1,d2):
    # stack the tensors along dimension 1 
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

def infer_scene_points(model, data_dir,run_octmae=False,set_conf=5,
               no_input_mask=False,no_pred_mask=False,no_filter_input_view=False,n_pred_views=5,
               do_filter_all_masks=False, dino_model=None):
    
    if dino_model is None:
        # Loading DINOv2 model
        dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14_reg")
        dino_model.eval()
        dino_model.to("cuda")

    dataloader_input_view = GenericLoaderSmall(data_dir,n_pred_views=1,pred_input_only=True,false_positive=None,false_negative=None)
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
        
        dataloader_pred_views = GenericLoaderSmall(data_dir,n_pred_views=n_pred_views,pred_input_only=False,
        pointmap_for_bb=input_points,run_octmae=run_octmae)
        pred_views_loader = DataLoader(dataloader_pred_views, batch_size=1, shuffle=True, collate_fn=collate)
        pred_views_batch = next(iter(pred_views_loader))

        pred_new_views, gt_new_views, _, scale_factor = model(pred_views_batch,dino_model)
        pred_new_views, pred_views_batch = postprocessor_pred_views(pred_new_views,pred_views_batch)
    
    pred = merge_dicts(dict_to_float(pred_input_view),dict_to_float(pred_new_views))
    gt = merge_dicts(dict_to_float(gt_input_view),dict_to_float(gt_new_views))

    batch = copy.deepcopy(input_view_batch)
    batch['new_cams'] = merge_dicts(input_view_batch['new_cams'],pred_views_batch['new_cams'])
    gt['pointmaps'] = None # make sure it's not used in viz
    
    if do_filter_all_masks:
        batch = filter_all_masks(pred,input_view_batch,max_outlier_views=1)

    # scale factor is the scale we applied to the input view for inference
    all_points = compute_all_points(pred,batch)
    all_points = all_points*(1.0/scale_factor)
    
    # transform all_points to the original coordinate system
    all_points_h = torch.cat([all_points,torch.ones(all_points.shape[:-1]+(1,)).to(all_points.device)],dim=-1)
    all_points_original = all_points_h @ batch['input_cams']['c2ws_original'][0][0].T
    all_points = all_points_original[...,:3]
    return all_points

def run_inference(data_directory):
    rayst3r_checkpoint = hf_hub_download("bartduis/rayst3r", "rayst3r.pth")
    
    model = Rayst3rModelWrapper(rayst3r_checkpoint,distributed=False)
    all_points = infer_scene_points(model, data_directory,run_octmae=False,set_conf=5,
                            no_input_mask=False,no_pred_mask=False,no_filter_input_view=False,n_pred_views=5,
                            do_filter_all_masks=False).cpu().numpy()
    all_points_save = os.path.join(data_directory,"inference_points.ply")
    o3d_pc = npy2ply(all_points,colors=None,normals=None)
    o3d.io.write_point_cloud(all_points_save, o3d_pc)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return o3d_pc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--run_octmae", action="store_true", default=False)
    parser.add_argument("--set_conf", type=float, default=5)
    parser.add_argument("--n_pred_views", type=int, default=5)
    parser.add_argument("--filter_all_masks", action="store_true", default=False)

    parser.add_argument("--no_input_mask", action="store_true", default=False)
    parser.add_argument("--no_pred_mask", action="store_true", default=False)
    parser.add_argument("--no_filter_input_view", action="store_true", default=False)
    args = parser.parse_args()
    
    print("Loading checkpoint from Huggingface")
    

if __name__ == "__main__":
    run_inference("/home/fabian/3d_reconstruction/zed_output/")