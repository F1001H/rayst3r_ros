import torch
import numpy as np
from PIL import Image
import os

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor

def detect_objects_with_owl(
    image_rgb_pil: Image.Image, 
    text_prompts: list,
    model: OwlViTForObjectDetection,
    processor: OwlViTProcessor,
    score_threshold: float = 0.1
):
    device = model.device
    if image_rgb_pil.mode != "RGB":
        image_rgb_pil = image_rgb_pil.convert("RGB")
        
    inputs = processor(text=[text_prompts], images=image_rgb_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image_rgb_pil.height, image_rgb_pil.width)]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=score_threshold,
        text_labels=[text_prompts] 
    )
    
    result = results[0] 
    boxes, scores, labels = result["boxes"], result["scores"], result["text_labels"]
    
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        detections.append({
            "box": [round(i, 2) for i in box.tolist()],
            "score": round(score.item(), 3),
            "label": label
        })
        
    return detections

def generate_mask_with_sam(image_rgb_cv2, bounding_box, sam_predictor): 
    input_box = np.array(bounding_box, dtype=np.float32)
    masks, scores, logits = sam_predictor.predict(
        box=input_box[None, :], 
        multimask_output=False  
    )
    return masks[0] 

def run_object_segmentation_pipeline(
    image_rgb_np: np.ndarray, 
    prompts_list: list,
    sam_checkpoint_path: str = "sam_models/sam_vit_h_4b8939.pth",
    owl_model_name: str = "google/owlvit-base-patch32",
    detection_threshold: float = 0.1,
    display_generated_mask: bool = True 
):
    if not prompts_list:
        print("Error: prompts_list cannot be empty.")
        return None 

    if not isinstance(image_rgb_np, np.ndarray) or image_rgb_np.ndim != 3 or image_rgb_np.shape[2] != 3:
        print("Error: image_rgb_np must be an HxWx3 NumPy array (RGB).")
        return None

    print("--- Loading segmentation models... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    owl_processor = None
    owl_model = None
    sam = None
    sam_predictor = None

    try:
        owl_processor = OwlViTProcessor.from_pretrained(owl_model_name)
        owl_model = OwlViTForObjectDetection.from_pretrained(owl_model_name).to(device)
        
        if not os.path.exists(sam_checkpoint_path):
            print(f"Error: SAM checkpoint '{sam_checkpoint_path}' not found.")
            return None 
        sam_model_type = "vit_h" 
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path).to(device)
        sam_predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Error loading models: {e}")
        return None 
    print(f"--- Segmentation models loaded successfully on {device}. ---")
    
    try:
        image_pil = Image.fromarray(image_rgb_np) 
        image_cv2_rgb_for_sam = image_rgb_np 
    except Exception as e:
        print(f"Error converting NumPy image: {e}")
        del owl_processor, owl_model, sam, sam_predictor
        if device == "cuda": torch.cuda.empty_cache()
        return None

    print(f"\n--- Step 1: Detecting objects for prompts: {prompts_list} ---")
    all_detections = detect_objects_with_owl(
        image_rgb_pil=image_pil, 
        text_prompts=prompts_list,
        model=owl_model,
        processor=owl_processor,
        score_threshold=detection_threshold
    )

    if not all_detections:
        print("Could not find any of the prompted objects in the image.")
        del owl_processor, owl_model, sam, sam_predictor
        if device == "cuda": torch.cuda.empty_cache()
        return np.zeros((image_rgb_np.shape[0], image_rgb_np.shape[1]), dtype=bool) 

    print(f"Found {len(all_detections)} objects.")
    for det in all_detections:
        print(f"  - Found '{det['label']}' with score {det['score']:.2f} at box {det['box']}")
    
    print(f"\n--- Step 2: Generating and combining masks... ---")
    
    height, width, _ = image_rgb_np.shape
    combined_mask_bool = np.zeros((height, width), dtype=bool) 

    sam_predictor.set_image(image_cv2_rgb_for_sam) 

    for detection in all_detections:
        box = detection['box'] 
        label = detection['label']
        print(f"  - Segmenting '{label}'...")
        
        try:
            individual_mask_bool = generate_mask_with_sam(image_cv2_rgb_for_sam, box, sam_predictor)
            combined_mask_bool = np.logical_or(combined_mask_bool, individual_mask_bool)
        except Exception as e:
            print(f"Error during SAM prediction for '{label}' with box {box}: {e}")
    
    print("All masks generated and combined successfully.")
    
    if display_generated_mask:
        print(f"\n--- Displaying combined mask (close window to continue)... ---")
        try:
            display_img_pil = Image.fromarray((combined_mask_bool.astype(np.uint8) * 255))
            display_img_pil.show()
        except Exception as e:
            print(f"Could not display mask: {e}")

    del owl_processor, owl_model, sam, sam_predictor, all_detections, image_pil
    if device == "cuda":
        torch.cuda.empty_cache()
    print("--- Segmentation model cleanup complete. ---")
    
    return combined_mask_bool