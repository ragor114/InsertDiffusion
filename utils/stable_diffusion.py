from diffusers import StableDiffusionUpscalePipeline, AutoPipelineForInpainting, AutoPipelineForImage2Image
import torch
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation, binary_fill_holes
import numpy as np
from .mask import get_mask_from_image
import random

# store pipelines globally to prevent loading them with every call
inpainting_pipeline = None
inpainting_model_name = None
img2img_pipeline = None
sd_model_name = None
upscaling_model_name = None
upscaling_pipeline = None
colorization_model_name = None
colorization_pipeline = None

# returns the number closest to n that is divisible by m, code from https://www.geeksforgeeks.org/find-number-closest-n-divisible-m/
def closest_divisible(n, m):
    q = n // m
    n1 = m*q
    if n*m > 0:
        n2 = (m * (q+1))
    else:
        n2 = (m * (q-1))
    
    if abs(n-n1) < abs(n-n2):
        return n1
    return n2

# load (SD2.1) inpainting pipeline
def get_inpainting_pipeline(model_name="stabilityai/stable-diffusion-2-inpainting", device="cuda" if torch.cuda.is_available() else 'cpu'):
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    return pipe

# load (SDXL) img2img pipeline
def get_img2img_pipeline(model_name="stabilityai/stable-diffusion-2-1", device="cuda" if torch.cuda.is_available() else 'cpu'):
    pipe = AutoPipelineForImage2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe = pipe.to(device)
    return pipe

# load stable diffusion upscaling pipeline for colorization
def get_upscaling_pipeline(model_name="stabilityai/stable-diffusion-x4-upscaler", device="cuda" if torch.cuda.is_available() else 'cpu'):
    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

# load (SDXL) inpainting pipeline for colorization
def get_colorization_pipeline(model_name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", device="cuda" if torch.cuda.is_available() else 'cpu'):
    pipe = AutoPipelineForInpainting.from_pretrained(model_name, torch_dtype=torch.float16, variant='fp16')
    pipe = pipe.to(device)
    return pipe

# this method implements inpainting and masked img2img generation
def sd_inpainting(model_name, image, mask_image, prompt=None, negative_prompt=None, prompt_guidance=7.5, total_steps=50, num_images: int=1, inpainting_strength: float=1):
    global inpainting_pipeline, inpainting_model_name

    # ensure correct data format
    image = image.convert("RGB")
    mask_image = mask_image.convert("RGB")

    # load inpainting pipeline if changed or first run
    if inpainting_pipeline is None or inpainting_model_name != model_name:
        inpainting_model_name = model_name
        inpainting_pipeline = get_inpainting_pipeline(model_name, "cuda" if torch.cuda.is_available() else "cpu")
    
    # ensure output has (roughly) same aspect ratio as input to not encounter "squashing" artefacts
    base_height = inpainting_pipeline.unet.config.sample_size * inpainting_pipeline.vae_scale_factor
    aspect = image.height / image.width
    new_height = int(aspect * base_height)
    new_height = closest_divisible(new_height, 8)

    # use inpainting pipeline
    inpainted_imgs = inpainting_pipeline(prompt=prompt, negative_prompt=negative_prompt, image=image, mask_image=mask_image, height=new_height, guidance_scale=prompt_guidance, num_inference_steps=total_steps, num_images_per_prompt=num_images, strength=inpainting_strength, generator=torch.Generator().manual_seed(random.randint(0, 1000000))).images

    # if only one image is requested then return this image, otherwise return the list of generated images
    if num_images == 1:
        return inpainted_imgs[0]
    return inpainted_imgs

# implements rediffusion as img2img diffusion
def sd_img2img(model_name, image, prompt=None, negative_prompt=None, strength=0.3, prompt_guidance=7.5, num_images: int=1):
    global img2img_pipeline, sd_model_name

    # ensure correct data format
    image = image.convert("RGB")
    
    # load img2img pipeline if changed or first run
    if img2img_pipeline is None or sd_model_name != model_name:
        sd_model_name = model_name
        img2img_pipeline = get_img2img_pipeline(model_name, "cuda" if torch.cuda.is_available() else "cpu")

    # ensure output has (roughly) same aspect ratio as input to not encounter "squashing" artefacts
    base_height = inpainting_pipeline.unet.config.sample_size * inpainting_pipeline.vae_scale_factor
    aspect = image.height / image.width
    new_height = int(aspect * base_height)
    new_height = closest_divisible(new_height, 8)
    
    # perform img2img diffusion
    results = img2img_pipeline(prompt=prompt, negative_prompt=negative_prompt, image=image, height=new_height, strength=strength, guidance_scale=prompt_guidance, num_images_per_prompt=num_images, generator=torch.Generator().manual_seed(random.randint(0, 1000000))).images
    
    # if only one image is requested then return this image, otherwise return the list of generated images
    if num_images == 1:
        return results[0]
    return results

# returns a binary mask that has True entry at every pixel that is non-white in the original image.
def get_colorization_mask(image: Image, mask_threshold: int=170, fill_holes: bool=True, dilation_iterations: int=10):
    # get mask of image based on threshold
    mask = get_mask_from_image(image, mask_threshold)
    # background -> foreground
    mask = ImageOps.invert(mask)
    
    # get binary mask
    mask_arr = np.array(mask)
    mask_arr = (mask_arr > 0)

    # dilation and filling holes improved performance
    if fill_holes:
        mask_arr = binary_fill_holes(mask_arr)
    mask_arr = binary_dilation(mask_arr, iterations=dilation_iterations)
    if fill_holes:
        mask_arr = binary_fill_holes(mask_arr)
    # get image from binary mask
    return Image.fromarray(mask_arr.astype(np.uint8)*255)

# this function performs colorization using masked Stable Diffusion img2img diffusion
def sd_colorization(col_model_name: str, up_model_name: str, low_res_image: Image, prompt: str, negative_prompt: str, mask_threshold=170, fill_holes: bool=True, dilation_iterations: int=10, prompt_guidance: float=10, colorization_steps: int=30, colorization_strength: float=0.9, num_images: int=1) -> Image:
    global colorization_model_name, colorization_pipeline, upscaling_model_name, upscaling_pipeline

    # load colorization model if changed or first run
    if colorization_model_name is None or colorization_model_name != col_model_name:
        colorization_model_name = col_model_name
        colorization_pipeline = get_colorization_pipeline(col_model_name, "cuda" if torch.cuda.is_available() else "cpu")

    # load upscaling model if changed or first run
    if upscaling_model_name is None or upscaling_model_name != up_model_name:
        upscaling_model_name = up_model_name
        upscaling_pipeline = get_upscaling_pipeline(up_model_name, "cuda" if torch.cuda.is_available() else "cpu")
    
    # first upscale image as we have observed that colorization performance is very poor on low resolution images
    if low_res_image.width < 750:
        upscaled_image = upscaling_pipeline(prompt, negative_prompt=negative_prompt, image=low_res_image, generator=torch.Generator().manual_seed(random.randint(0, 1000000))).images[0]
    else:
        upscaled_image = low_res_image
    # get mask for image
    mask = get_colorization_mask(upscaled_image, mask_threshold, fill_holes, dilation_iterations)
    # perform masked img2img diffusion
    colorized_images = colorization_pipeline(prompt, negative_prompt=negative_prompt, image=upscaled_image, mask_image=mask, guidance_scale=prompt_guidance, num_inference_steps=colorization_steps, strength=colorization_strength, num_images_per_prompt=num_images, generator=torch.Generator().manual_seed(random.randint(0, 1000000))).images

    # if only one image is requested then return this image, otherwise return the list of generated images
    if num_images == 1:
        return colorized_images[0]
    return colorized_images