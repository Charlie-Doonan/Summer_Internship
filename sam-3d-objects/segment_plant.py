import sys
import numpy as np
from PIL import Image

# Import SAM3D inference code
sys.path.append("notebook")

from inference import Inference

CONFIG = "checkpoints/hf/pipeline.yaml"

IMAGE_PATH = "plant.jpg"
MASK_PATH = "plant_mask.jpg"

print("Loading model...")
inference = Inference(CONFIG, compile=False)

print("Loading image...")

# Force RGB (SAM3D expects H,W,3)
image = np.array(
    Image.open(IMAGE_PATH).convert("RGB"),
    dtype=np.uint8
)

# Force single-channel binary mask (SAM3D expects H,W)
mask = np.array(
    Image.open(MASK_PATH).convert("L"),
    dtype=np.uint8
)

mask = mask > 0

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

print("Running reconstruction...")

output = inference(
    image,
    mask,
    seed=42
)

print("Saving output...")
output["gs"].save_ply("plant_splat.ply")
print("Saved plant_splat.ply")