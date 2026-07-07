import torch
from pathlib import Path
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = Sam3Model.from_pretrained("./checkpoints").to(device)
processor = Sam3Processor.from_pretrained("./checkpoints")

# Assets folder
assets_dir = Path("assets")

# Input and output files
input_path = assets_dir / "plant.jpg"
output_path = assets_dir / "stem_mask.jpg"

# Load image
image = Image.open(input_path).convert("RGB")

# Run segmentation
inputs = processor(
    images=image,
    text="plant stem",
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

masks = results["masks"]

if len(masks) == 0:
    print("No plant stem detected.")
    image.save(output_path)
else:
    image_np = np.array(image)
    combined = np.zeros(image_np.shape[:2], dtype=bool)

    for m in masks:
        combined |= m.cpu().numpy().astype(bool)

    out = np.zeros_like(image_np)
    out[combined] = image_np[combined]

    Image.fromarray(out).save(output_path)
    print(f"Saved segmented stem to: {output_path}")