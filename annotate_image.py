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


if __name__ == "__main__":
    # --- Configuration ---
    IMAGE_PATH = "zed_output/rgb.png"
    # *** NEW: Provide a list of all objects you want to find ***
    PROMPTS = ["a pringles can", "a stool", "a laptop screen"]
    
    SAM_CHECKPOINT_PATH = "sam_models/sam_vit_h_4b8939.pth"
    OWL_MODEL_ID = "google/owlvit-base-patch32"
    
    OUTPUT_MASK_PATH = "zed_output/mask.png"
    
    # --- 1. Setup Models (Load them once) ---
    print("--- Loading models (this may take a moment)... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    owl_processor = OwlViTProcessor.from_pretrained(OWL_MODEL_ID)
    owl_model = OwlViTForObjectDetection.from_pretrained(OWL_MODEL_ID).to(device)
    
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device)
    sam_predictor = SamPredictor(sam)
    
    print("--- Models loaded successfully. ---")
    
    # --- 2. Step 1: Detect ALL Objects with Owl-ViT ---
    print(f"\n--- Step 1: Detecting objects for prompts: {PROMPTS} ---")
    all_detections = detect_objects_in_image(
        image_path=IMAGE_PATH,
        text_prompts=PROMPTS,
        model=owl_model,
        processor=owl_processor,
        score_threshold=0.1
    )

    if not all_detections:
        print("Could not find any of the prompted objects in the image. Exiting.")
        sys.exit(1)
        
    print(f"Found {len(all_detections)} objects.")
    for det in all_detections:
        print(f"  - Found '{det['label']}' with score {det['score']:.2f}")
    
    # --- 3. Step 2: Generate and Combine Masks for Each Object ---
    print(f"\n--- Step 2: Generating and combining masks for all detected objects... ---")
    
    image_cv2 = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    
    # Initialize an empty boolean mask with the same dimensions as the image
    height, width, _ = image_rgb.shape
    combined_mask = np.zeros((height, width), dtype=bool)

    # Loop through each detection
    for detection in all_detections:
        box = detection['box']
        label = detection['label']
        print(f"  - Segmenting '{label}'...")
        
        # Generate the individual mask for the current object
        individual_mask = generate_mask_with_sam(image_rgb, box, sam_predictor)
        
        # Combine the individual mask with the total mask using a logical OR.
        # This adds the new object's pixels to the final mask.
        combined_mask = np.logical_or(combined_mask, individual_mask)
    
    print("All masks generated and combined successfully.")

    # --- 4. Step 3: Save the Combined Binary Mask ---
    print(f"\n--- Step 3: Saving combined binary mask to '{OUTPUT_MASK_PATH}'... ---")
    
    # Convert the final boolean mask to a binary image (0/255)
    binary_mask_image = (combined_mask.astype(np.uint8) * 255)
    
    Image.fromarray(binary_mask_image).save(OUTPUT_MASK_PATH)
    print("Mask saved.")
    
    # Optional: Display the final combined mask
    try:
        Image.fromarray(binary_mask_image).show()
    except Exception as e:
        print(f"Could not display mask: {e}")