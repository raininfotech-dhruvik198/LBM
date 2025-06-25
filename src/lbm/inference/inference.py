import logging

import PIL
import torch
from torchvision.transforms import ToPILImage, ToTensor

from lbm.models.lbm import LBMModel
from lbm.inference.utils import resize_and_center_crop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@torch.no_grad()
def evaluate(
    model: LBMModel,
    source_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
):
    """
    Evaluate the model on an image coming from the source distribution and generate a new image from the target distribution.

    Args:
        model (LBMModel): The model to evaluate.
        source_image (PIL.Image.Image): The source image to evaluate the model on.
        num_sampling_steps (int): The number of sampling steps to use for the model.

    Returns:
        PIL.Image.Image: The generated image.
    """

    ori_w, ori_h = source_image.size

    target_w = int(round(ori_w / 32)) * 32
    target_h = int(round(ori_h / 32)) * 32

    source_image = resize_and_center_crop(source_image, target_w, target_h)

    img_pasted_tensor = ToTensor()(source_image).unsqueeze(0) * 2 - 1
    batch = {
        "source_image": img_pasted_tensor.cuda().to(torch.bfloat16),
    }

    z_source = model.vae.encode(batch[model.source_key])

    output_image = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    output_image = (output_image[0].float().cpu() + 1) / 2
    output_image = ToPILImage()(output_image)
    output_image = resize_and_center_crop(output_image, ori_w, ori_h)

    return output_image
