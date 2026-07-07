import cv2
import os
import argparse
import numpy as np

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# Relative paths (run from the segment-anything directory)
DEFAULT_CHECKPOINT = os.path.join("checkpoint", "sam_vit_b_01ec64.pth")
DEFAULT_MODEL_TYPE = "vit_b"
DEFAULT_INPUT = os.path.join("assets", "plant.jpg")
DEFAULT_OUTPUT = os.path.join("assets", "plant_segmented.png")
DEFAULT_DEVICE = "cuda"


def get_green_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 35, 35])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask > 0


def mask_score(mask, green_mask):
    seg = mask["segmentation"]
    overlap = np.logical_and(seg, green_mask).sum()
    area = seg.sum() + 1e-6
    return overlap / area


def merge_masks(masks, green_mask, threshold=0.12):
    combined = np.zeros_like(green_mask, dtype=np.uint8)

    for m in masks:
        score = mask_score(m, green_mask)
        if score > threshold:
            combined = np.logical_or(combined, m["segmentation"])

    return combined.astype(np.uint8)


def clean_mask(mask):
    mask = mask.astype(np.uint8) * 255
    kernel_small = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    mask = cv2.dilate(mask, kernel_large, iterations=1)
    return mask > 0


def apply_mask(image, mask):
    out = np.zeros_like(image)
    out[mask] = image[mask]
    return out


def main(args):
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)

    generator = SamAutomaticMaskGenerator(sam)

    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise ValueError(f"Could not load image: {args.input}")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    green_mask = get_green_mask(image)
    masks = generator.generate(image)

    merged = merge_masks(masks, green_mask, threshold=0.12)
    final_mask = clean_mask(merged)

    result = apply_mask(image, final_mask)

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, result_bgr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    args = parser.parse_args()
    main(args)