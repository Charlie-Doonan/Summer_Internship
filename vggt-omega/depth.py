import torch
import cv2
import numpy as np

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

checkpoint_path = "checkpoints/vggt_omega_1b_512.pt"
image_names = ["plant.jpg"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VGGTOmega().to(device).eval()
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

images = load_and_preprocess_images(image_names, image_resolution=512).to(device)

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_img = depth.squeeze().detach().cpu().numpy()

depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
depth_img = depth_img.astype(np.uint8)

depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

cv2.imwrite("depth.png", depth_colored)