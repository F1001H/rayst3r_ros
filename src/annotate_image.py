import torch
import numpy as np
from PIL import Image
import os

# --- Model Imports ---
# For YOLO-World
from ultralytics import YOLOWorld
# For OWL-ViT
from transformers import OwlViTProcessor, OwlViTForObjectDetection
# For SAM
from segment_anything import sam_model_registry, SamPredictor

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +                            DETECTION FUNCTIONS                            +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def detect_objects_with_yolo_world(
    image_rgb_pil: Image.Image,
    text_prompts: list,
    model: YOLOWorld,
    score_threshold: float = 0.01
) -> list:
    """Detects objects using YOLO-World."""
    if not text_prompts:
        return []
    model.set_classes(text_prompts)
    results = model.predict(image_rgb_pil, conf=score_threshold, verbose=False)
    detections = []
    if results and results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            detections.append({
                "box": [round(i, 2) for i in box.xyxy[0].tolist()],
                "score": round(box.conf[0].item(), 3),
                "label": text_prompts[class_id]
            })
    return detections

def detect_objects_with_owl(
    image_rgb_pil: Image.Image,
    text_prompts: list,
    model: OwlViTForObjectDetection,
    processor: OwlViTProcessor,
    score_threshold: float = 0.01
) -> list:
    """Detects objects using OWL-ViT."""
    device = model.device
    if image_rgb_pil.mode != "RGB":
        image_rgb_pil = image_rgb_pil.convert("RGB")

    inputs = processor(text=text_prompts, images=image_rgb_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_rgb_pil.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=score_threshold
    )
    
    # The new post_process_object_detection returns a list of dicts per image
    result = results[0]
    boxes, scores, labels = result["boxes"], result["scores"], result["labels"]

    detections = []
    for box, score, label_id in zip(boxes, scores, labels):
        detections.append({
            "box": [round(i, 2) for i in box.tolist()],
            "score": round(score.item(), 3),
            "label": text_prompts[label_id]
        })
    return detections

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +                          SEGMENTATION FUNCTIONS                           +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def generate_mask_with_sam(image_rgb_np: np.ndarray, bounding_box: list, sam_predictor: SamPredictor) -> np.ndarray:
    """Generates a mask for a given bounding box using SAM."""
    input_box = np.array(bounding_box, dtype=np.float32)
    masks, _, _ = sam_predictor.predict(box=input_box[None, :], multimask_output=False)
    return masks[0]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +                              MAIN PIPELINE                              +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def run_object_segmentation_pipeline(
    image_rgb_np: np.ndarray,
    prompts_list: list,
    detector_type: str = "yolo-world", # <-- KEY VARIABLE TO CHOOSE THE MODEL
    sam_checkpoint_path: str = "sam_models/sam_vit_h_4b8939.pth",
    yolo_model_name: str = "yolov8x-worldv2.pt",
    owl_model_name: str = "google/owlvit-base-patch32",
    detection_threshold: float = 0.25,
    display_generated_mask: bool = False
) -> tuple[list[np.ndarray], list[tuple]]:
    """
    Detects and segments objects using a chosen detector (YOLO-World or OWL-ViT) and SAM.
    
    Args:
        image_rgb_np (np.ndarray): Input image as an HxWx3 NumPy array (RGB).
        prompts_list (list): Text prompts for objects to detect.
        detector_type (str): The object detector to use. Either "yolo-world" or "owl-vit".
        sam_checkpoint_path (str): Path to the SAM checkpoint file.
        yolo_model_name (str): Name of the YOLO-World model.
        owl_model_name (str): Name of the OwlViT model on Hugging Face Hub.
        detection_threshold (float): Confidence score threshold for detection.
        display_generated_mask (bool): If True, displays each generated mask.

    Returns:
        A tuple of two lists: (list_of_masks, list_of_bounding_boxes).
    """
    if not prompts_list:
        print("Error: prompts_list cannot be empty.")
        return [], []
    if not isinstance(image_rgb_np, np.ndarray) or image_rgb_np.ndim != 3 or image_rgb_np.shape[2] != 3:
        print("Error: image_rgb_np must be an HxWx3 NumPy array (RGB).")
        return [], []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running on device: {device} ---")
    print(f"--- Using detector: {detector_type} ---")

    # --- Load Models ---
    detector_model, owl_processor, sam, sam_predictor = None, None, None, None
    try:
        # Load the chosen detector
        if detector_type.lower() == "yolo-world":
            detector_model = YOLOWorld(yolo_model_name)
        elif detector_type.lower() == "owl-vit":
            owl_processor = OwlViTProcessor.from_pretrained(owl_model_name)
            detector_model = OwlViTForObjectDetection.from_pretrained(owl_model_name).to(device)
        else:
            raise ValueError(f"Invalid detector_type '{detector_type}'. Choose 'yolo-world' or 'owl-vit'.")

        # Load SAM (common for both detectors)
        if not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint '{sam_checkpoint_path}' not found.")
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to(device)
        sam_predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Error loading models: {e}")
        return [], []
    print("--- Models loaded successfully. ---")

    image_pil = Image.fromarray(image_rgb_np)

    # --- Step 1: Object Detection ---
    print(f"\n--- Detecting objects for prompts: {prompts_list} ---")
    all_detections = []
    if detector_type.lower() == "yolo-world":
        all_detections = detect_objects_with_yolo_world(
            image_pil, prompts_list, detector_model, detection_threshold
        )
    elif detector_type.lower() == "owl-vit":
        all_detections = detect_objects_with_owl(
            image_pil, prompts_list, detector_model, owl_processor, detection_threshold
        )

    if not all_detections:
        print("Could not find any of the prompted objects in the image.")
        return [], []

    print(f"Found {len(all_detections)} objects.")
    for det in all_detections:
        print(f"  - Found '{det['label']}' with score {det['score']:.2f} at box {det['box']}")

    # --- Step 2: Mask Generation with SAM ---
    print(f"\n--- Generating masks for detected objects... ---")
    all_masks_list = []
    all_boxes_list = []
    sam_predictor.set_image(image_rgb_np)

    for detection in all_detections:
        box_corners, label = detection['box'], detection['label']
        print(f"  - Segmenting '{label}'...")
        try:
            mask = generate_mask_with_sam(image_rgb_np, box_corners, sam_predictor)
            all_masks_list.append(mask)

            xmin, ymin, xmax, ymax = box_corners
            center_based_box = (xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2, xmax - xmin, ymax - ymin)
            all_boxes_list.append({
            "box": center_based_box,
            "label": label,            
            })
            if display_generated_mask:
                Image.fromarray((mask.astype(np.uint8) * 255)).show()
        except Exception as e:
            print(f"Error during SAM prediction for '{label}': {e}")

    print("--- Segmentation complete. ---")
    
    # --- Cleanup ---
    del detector_model, sam, sam_predictor, all_detections, image_pil
    if owl_processor:
        del owl_processor
    if device == "cuda":
        torch.cuda.empty_cache()
    print("--- Model cleanup complete. ---")

    return all_masks_list, all_boxes_list

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +                                 EXAMPLE                                   +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
    # Create a dummy image for demonstration (e.g., a 600x800 image with 3 channels)
    # In a real scenario, you would load your image here:
    # from PIL import Image
    # image = Image.open("path/to/your/image.jpg").convert("RGB")
    # image_rgb_np = np.array(image)
    
    # Dummy image: a blue square (person) and a red circle (a stop sign) on a gray background
    print("Creating a dummy image for demonstration...")
    image_rgb_np = np.full((600, 800, 3), 128, dtype=np.uint8)
    # Blue square (person)
    image_rgb_np[100:300, 150:250] = [0, 0, 255] 
    # Red rectangle (bus)
    image_rgb_np[350:550, 400:700] = [255, 0, 0]
    
    my_prompts = ["a person", "a red bus"]

    # --- IMPORTANT ---
    # You must download the SAM checkpoint file first.
    # e.g., from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    sam_checkpoint = "sam_vit_h_4b8939.pth"

    if not os.path.exists(sam_checkpoint):
        print("\nERROR: SAM Checkpoint not found!")
        print(f"Please download '{sam_checkpoint}' and place it in the same directory.")
    else:
        # ---== Run pipeline with YOLO-World ==---
        print("\n" + "="*50)
        print("RUNNING PIPELINE WITH YOLO-WORLD")
        print("="*50)
        masks_yolo, boxes_yolo = run_object_segmentation_pipeline(
            image_rgb_np,
            my_prompts,
            detector_type="yolo-world", # Specify yolo-world
            sam_checkpoint_path=sam_checkpoint
        )
        print(f"YOLO-World found {len(masks_yolo)} objects.")

        # ---== Run pipeline with OWL-ViT ==---
        print("\n" + "="*50)
        print("RUNNING PIPELINE WITH OWL-ViT")
        print("="*50)
        masks_owl, boxes_owl = run_object_segmentation_pipeline(
            image_rgb_np,
            my_prompts,
            detector_type="owl-vit", # Specify owl-vit
            sam_checkpoint_path=sam_checkpoint
        )
        print(f"OWL-ViT found {len(masks_owl)} objects.")