import os
import torch
from PIL import Image, ImageOps
from cog import BasePredictor, Input, Path
import tempfile
import numpy as np # Added for array conversion
from torchvision import transforms # Added for utility functions
from transformers import AutoModelForImageSegmentation # Added for BiRefNet
from copy import deepcopy # Added, was used in gradio_demo

# Assuming 'lbm' is in 'src' and 'src' is in PYTHONPATH or added to it.
# If 'src' is not automatically in PYTHONPATH in the Cog environment,
# we might need to add:
# import sys
# sys.path.append('src')

import sys
import os
# Add the directory containing this script (which is /) to sys.path
# so that the 'lbm' module (located in /src/lbm) can be found.
# __file__ is /predict.py, so os.path.dirname(__file__) is /
# To access /src/lbm, we need to add /src to sys.path.
# Assuming the script is run from the root of the project,
# 'src' should be directly accessible.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from lbm.inference import get_model # Still need get_model for LBM

# --- Utility functions (copied from examples/inference/utils.py and adapted) ---
def extract_object(birefnet_model, img: Image.Image):
    # Data settings for BiRefNet
    image_size = (1024, 1024) # BiRefNet default input size
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    original_image_size = img.size
    # Ensure image is RGB for transformations
    input_image_for_birefnet = img.convert("RGB")
    input_tensor = transform_image(input_image_for_birefnet).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Prediction
    with torch.no_grad():
        preds = birefnet_model(input_tensor)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    mask_pil = transforms.ToPILImage()(pred)

    # Resize mask to original image size
    mask = mask_pil.resize(original_image_size, Image.NEAREST) # Use NEAREST for masks

    # The original util also returned a composite of image with gray background,
    # but for Cog we primarily need the mask.
    # image_composite = Image.composite(img, Image.new("RGB", img.size, (127, 127, 127)), mask.convert('L'))
    return mask # Return only the mask, ensure it's L mode for Image.composite

def resize_and_center_crop(image: Image.Image, target_width: int, target_height: int):
    original_width, original_height = image.size
    if original_width == target_width and original_height == target_height:
        return image

    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))

    resized_image = image.resize((resized_width, resized_height), Image.LANCZOS)

    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2

    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image

# --- Aspect Ratios (from gradio_demo.py) ---
ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048), str(1024 / 1024): (1024, 1024), str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152), str(1152 / 896): (1152, 896), str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536), str(768 / 1280): (768, 1280), str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640), str(1920 / 512): (1920, 512),
}

# Define a cache directory for models within the Cog environment
MODEL_CACHE_DIR = "ckpts"

class Predictor(BasePredictor):
    def setup(self):
        """Load models into memory."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        # Load LBM Relighting Model
        lbm_model_name = "relighting"
        lbm_model_path = os.path.join(MODEL_CACHE_DIR, lbm_model_name)
        if not os.path.exists(os.path.join(lbm_model_path, "unet")): # Check for a subfile/dir to confirm full download
            print(f"Downloading LBM {lbm_model_name} model from HF hub to {lbm_model_path}...")
            os.makedirs(lbm_model_path, exist_ok=True)
            self.lbm_model = get_model(
                f"jasperai/LBM_{lbm_model_name}",
                save_dir=lbm_model_path,
                torch_dtype=torch.bfloat16,
                device=self.device,
            )
        else:
            print(f"Loading LBM {lbm_model_name} model from local cache: {lbm_model_path}...")
            self.lbm_model = get_model(
                lbm_model_path,
                torch_dtype=torch.bfloat16,
                device=self.device,
            )

        # Load BiRefNet Segmentation Model
        birefnet_cache_path = os.path.join(MODEL_CACHE_DIR, "birefnet_cache")
        os.makedirs(birefnet_cache_path, exist_ok=True)
        print("Loading BiRefNet segmentation model...")
        self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True,
            cache_dir=birefnet_cache_path # Use a cache dir for HF models
        ).to(self.device)
        self.birefnet_model.eval() # Set to eval mode

        # For ToTensor and ToPILImage, which don't have state
        self.to_tensor = transforms.ToTensor()
        self.to_pil_image = transforms.ToPILImage()

        print("All models loaded.")

    def predict(
        self,
        foreground_image: Path = Input(description="Foreground image with the object."),
        background_image: Path = Input(description="Background image for lighting context."),
        num_sampling_steps: int = Input(
            description="Number of inference steps for LBM model.", default=1, ge=1, le=4
        ),
        output_width: int = Input(description="Optional. Desired width for the final output image. If not provided, defaults to original foreground width.", default=None, ge=1),
        output_height: int = Input(description="Optional. Desired height for the final output image. If not provided, defaults to original foreground height.", default=None, ge=1)
    ) -> Path:
        """Run relighting prediction with foreground and background images."""

        fg_image_pil = Image.open(str(foreground_image)).convert("RGB")
        bg_image_pil = Image.open(str(background_image)).convert("RGB")

        # --- Logic from gradio_demo.py's evaluate function ---
        ori_w_fg, ori_h_fg = fg_image_pil.size # Corrected: PIL uses (width, height)
        ar_fg = ori_h_fg / ori_w_fg # Aspect ratio based on height/width

        # Find closest aspect ratio for processing dimensions
        # Note: gradio_demo used fg_image.size for this, which seems more robust
        closest_ar_key = min(ASPECT_RATIOS.keys(), key=lambda x: abs(float(x) - ar_fg))
        dimensions_processing = ASPECT_RATIOS[closest_ar_key]

        # Extract foreground mask using BiRefNet
        # The deepcopy was used in gradio, good practice if original fg_image_pil is needed later
        fg_mask_pil = extract_object(self.birefnet_model, deepcopy(fg_image_pil))
        # Ensure mask is 'L' mode for Image.composite
        if fg_mask_pil.mode != 'L':
            fg_mask_pil = fg_mask_pil.convert('L')


        # Resize and crop images and mask
        # Target dimensions for processing (h, w)
        proc_h, proc_w = dimensions_processing

        fg_image_processed = resize_and_center_crop(fg_image_pil, proc_w, proc_h)
        # Make sure mask is also L mode before resize_and_center_crop if it expects RGB
        # The resize_and_center_crop in utils.py doesn't specify mode, assumes PIL image.
        # Mask should be single channel.
        fg_mask_processed = resize_and_center_crop(fg_mask_pil, proc_w, proc_h)
        bg_image_processed = resize_and_center_crop(bg_image_pil, proc_w, proc_h)

        # Composite fg on bg for LBM input
        img_pasted = Image.composite(fg_image_processed, bg_image_processed, fg_mask_processed)

        # Prepare batch for LBM model
        # Normalize to [-1, 1]
        img_pasted_tensor = (self.to_tensor(img_pasted).unsqueeze(0) * 2 - 1).to(self.device).to(self.lbm_model.dtype)

        batch = {
            self.lbm_model.source_key: img_pasted_tensor,
            # Add other conditioning inputs if LBM_relighting model expects them
            # Based on gradio demo, it seems source_image is the primary input from z_source
        }

        # LBM model inference (adapted from gradio_demo.py)
        # The LBM model in gradio uses model.vae.encode and model.sample
        # The get_model might return a wrapper; let's check how lbm.inference.evaluate does it
        # The original lbm.inference.evaluate uses:
        #   z_source = model.encode(batch[model.source_key])
        #   outputs = model.decode(z_source, **kwargs) -> This is for LBM_normals, LBM_depth
        # The gradio demo for relighting uses:
        #   z_source = model.vae.encode(batch[model.source_key])
        #   output_image = model.sample(z=z_source, num_steps=num_sampling_steps, conditioner_inputs=batch, max_samples=1)
        # This suggests the 'relighting' model variant might have a different inference path or structure.
        # For now, sticking to the gradio_demo.py's direct use of .vae.encode and .sample for relighting.

        print(f"Encoding source image...")
        z_source = self.lbm_model.vae.encode(batch[self.lbm_model.source_key])

        print(f"Running LBM sampling with {num_sampling_steps} steps...")
        lbm_output_tensor = self.lbm_model.sample(
            z=z_source,
            num_steps=num_sampling_steps,
            conditioner_inputs=batch, # Pass the whole batch for potential conditioning
            max_samples=1,
        ).clamp(-1, 1)

        # Denormalize from [-1, 1] to [0, 1] and convert to PIL Image
        lbm_output_image_pil = self.to_pil_image((lbm_output_tensor[0].float().cpu() + 1) / 2)

        # Composite LBM output back onto the processed background using the processed mask
        final_output_image_processed = Image.composite(lbm_output_image_pil, bg_image_processed, fg_mask_processed)

        # Resize to original foreground image dimensions
        if output_width is not None and output_height is not None:
            print(f"Resizing final output to user-defined dimensions: {output_width}x{output_height} using resize_and_center_crop.")
            final_output_image_resized = resize_and_center_crop(final_output_image_processed, output_width, output_height)
        else:
            print(f"Resizing final output to original foreground dimensions: {ori_w_fg}x{ori_h_fg} using simple resize.")
            final_output_image_resized = final_output_image_processed.resize((ori_w_fg, ori_h_fg), Image.LANCZOS)

        # Save the output image
        out_dir = tempfile.mkdtemp()
        out_path = Path(out_dir) / "output.png"
        final_output_image_resized.save(out_path)

        print(f"Output image saved to: {out_path}")
        return out_path
