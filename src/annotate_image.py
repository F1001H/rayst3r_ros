import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2 # Using OpenCV for image handling compatible with SAM
import os
import sys

# --- Model-specific imports ---
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor

# ===================================================================
#  Part 1: Modified Owl-ViT Detection Function
# ===================================================================
def detect_objects_in_image(
    image_path: str,
    text_prompts: list,
    model: OwlViTForObjectDetection,
    processor: OwlViTProcessor,
    score_threshold: float = 0.1
):
    """
    Detects ALL objects matching the text prompts and returns them as a list.
    """
    device = model.device
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text_prompts], images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)]).to(device)
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
        
    # *** KEY CHANGE: Return all detections, not just the best one. ***
    return detections


# ===================================================================
#  Part 2: Unchanged SAM Mask Generation Function
# ===================================================================
def generate_mask_with_sam(image_rgb, bounding_box, sam_predictor):
    """
    Generates a segmentation mask for a given bounding box using SAM.
    This function is unchanged but will be called in a loop.
    """
    input_box = np.array(bounding_box, dtype=np.float32)

    # The image is set only once per session, but calling it again is safe.
    sam_predictor.set_image(image_rgb)

    masks, scores, logits = sam_predictor.predict(
        box=input_box[None, :],
        multimask_output=False
    )
    
    return masks[0]

def run_object_segmentation_pipeline(
    data_directory_path: str,
    prompts_list: list = None,
    image_filename: str = "rgb.png",
    output_mask_filename: str = "mask.png",
    sam_checkpoint: str = "sam_models/sam_vit_h_4b8939.pth",
    owl_model_name: str = "google/owlvit-base-patch32",
    detection_threshold: float = 0.1,
    display_final_mask: bool = False
):
    if prompts_list is None:
        print("Error: Prompt list is empty")

    image_path = os.path.join(data_directory_path, image_filename)
    output_mask_path = os.path.join(data_directory_path, output_mask_filename)

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return False
    
    if not os.path.isdir(os.path.dirname(output_mask_path)): # Ensure output dir exists
        try:
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        except OSError as e:
            print(f"Error creating directory for output mask {output_mask_path}: {e}")
            return False


    # --- 1. Setup Models (Load them once per call to this function) ---
    print(f"--- Processing: {data_directory_path} ---")
    print("--- Loading models (this may take a moment)... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        owl_processor = OwlViTProcessor.from_pretrained(owl_model_name)
        owl_model = OwlViTForObjectDetection.from_pretrained(owl_model_name).to(device)
        
        if not os.path.exists(sam_checkpoint):
            print(f"Error: SAM checkpoint '{sam_checkpoint}' not found.")
            return False
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
        sam_predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Error loading models: {e}")
        return False
    print(f"--- Models loaded successfully on {device}. ---")
    
    # --- 2. Step 1: Detect ALL Objects with Owl-ViT ---
    print(f"\n--- Step 1: Detecting objects for prompts: {prompts_list} ---")
    all_detections = detect_objects_in_image( # Uses the existing helper function
        image_path=image_path,
        text_prompts=prompts_list,
        model=owl_model,
        processor=owl_processor,
        score_threshold=detection_threshold
    )

    if not all_detections:
        print("Could not find any of the prompted objects in the image.")
        del owl_processor, owl_model, sam, sam_predictor # Cleanup
        if device == "cuda": torch.cuda.empty_cache()
        return False # Indicate failure or no detections
        
    print(f"Found {len(all_detections)} objects.")
    for det in all_detections:
        print(f"  - Found '{det['label']}' with score {det['score']:.2f}")
    
    # --- 3. Step 2: Generate and Combine Masks for Each Object ---
    print(f"\n--- Step 2: Generating and combining masks for all detected objects... ---")
    
    try:
        image_cv2 = cv2.imread(image_path)
        if image_cv2 is None:
            print(f"Error reading image {image_path} with OpenCV.")
            del owl_processor, owl_model, sam, sam_predictor; 
            if device == "cuda": torch.cuda.empty_cache()
            return False
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error processing image {image_path} with OpenCV: {e}")
        del owl_processor, owl_model, sam, sam_predictor; 
        if device == "cuda": torch.cuda.empty_cache()
        return False

    height, width, _ = image_rgb.shape
    combined_mask = np.zeros((height, width), dtype=bool)

    for detection in all_detections:
        box = detection['box']
        label = detection['label']
        print(f"  - Segmenting '{label}'...")
        
        try:
            individual_mask = generate_mask_with_sam(image_rgb, box, sam_predictor) # Uses existing helper
            combined_mask = np.logical_or(combined_mask, individual_mask)
        except Exception as e:
            print(f"Error during SAM prediction for '{label}' with box {box}: {e}")
            # Decide if you want to continue or fail
            # continue
    
    print("All masks generated and combined successfully.")

    # --- 4. Step 3: Save the Combined Binary Mask ---
    binary_mask_image_pil = None # For potential display
    print(f"\n--- Step 3: Saving combined binary mask to '{output_mask_path}'... ---")
    try:
        binary_mask_to_save = (combined_mask.astype(np.uint8) * 255)
        binary_mask_image_pil = Image.fromarray(binary_mask_to_save)
        binary_mask_image_pil.save(output_mask_path)
        print("Mask saved.")
    except Exception as e:
        print(f"Error saving mask to {output_mask_path}: {e}")
        # Decide if this is a fatal error for the function's success
    
    # Optional: Display the final combined mask
    if display_final_mask and binary_mask_image_pil is not None:
        print(f"\n--- Displaying combined mask (close window to continue)... ---")
        try:
            binary_mask_image_pil.show()
        except Exception as e:
            print(f"Could not display mask: {e}")

    # Cleanup models
    del owl_processor, owl_model, sam, sam_predictor, all_detections
    if device == "cuda":
        torch.cuda.empty_cache()
    print("--- Model cleanup complete. ---")
    
    return True