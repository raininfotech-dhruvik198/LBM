import os
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
import tempfile

# It's good practice to ensure imports from the project are robust.
# Assuming 'lbm' is in 'src' and 'src' is in PYTHONPATH or added to it.
# If 'src' is not automatically in PYTHONPATH in the Cog environment,
# we might need to add:
# import sys
# sys.path.append('src')
from lbm.inference import get_model, evaluate

# Define model choices based on the inference script
MODEL_CHOICES = ["normals", "depth", "relighting"]
DEFAULT_MODEL_NAME = "normals"
# Define a cache directory for models within the Cog environment
MODEL_CACHE_DIR = "ckpts"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running predictions faster."""
        self.models = {}
        # Ensure the model cache directory exists
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        for model_name in MODEL_CHOICES:
            model_path = os.path.join(MODEL_CACHE_DIR, model_name)
            if not os.path.exists(model_path):
                print(f"Downloading {model_name} LBM model from HF hub to {model_path}...")
                # Ensure the specific model sub-directory is created before saving
                os.makedirs(model_path, exist_ok=True)
                self.models[model_name] = get_model(
                    f"jasperai/LBM_{model_name}",
                    save_dir=model_path,
                    torch_dtype=torch.bfloat16, # Consider torch.float16 if bfloat16 is not supported or for wider compatibility
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            else:
                print(f"Loading {model_name} LBM model from local cache: {model_path}...")
                self.models[model_name] = get_model(
                    model_path,
                    torch_dtype=torch.bfloat16, # Consider torch.float16
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
        print("All models loaded.")

    def predict(
        self,
        source_image: Path = Input(description="Source image for processing."),
        model_name: str = Input(
            description="Choose the model for processing.",
            choices=MODEL_CHOICES,
            default=DEFAULT_MODEL_NAME,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps.", default=1, ge=1, le=100 # Assuming a reasonable range
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {', '.join(MODEL_CHOICES)}")

        print(f"Processing with model: {model_name}")
        model = self.models[model_name]

        img = Image.open(str(source_image)).convert("RGB")

        print(f"Running evaluation with {num_inference_steps} steps...")
        output_image = evaluate(model, img, num_inference_steps)

        # Save the output image to a temporary file
        out_dir = tempfile.mkdtemp()
        out_path = Path(out_dir) / "output.png"
        output_image.save(out_path)

        print(f"Output image saved to: {out_path}")
        return out_path
