import argparse
import os
from PIL import Image
import torch
from datetime import datetime
from utils import get_mask_and_background, paste_pipeline, get_mask_from_image, sd_inpainting, sd_img2img, sd_colorization, get_dataframe_row, paste_image, extract_object
from utils import get_bike_colorization_prompt, get_bike_inpainting_prompt, get_car_colorization_prompt, get_car_inpainting_prompt, get_product_colorization_prompt, get_product_inpainting_prompt
from utils import inpaint as bike_inpainting

# this function can be used to generate a bike based on reference points and wheels using a pretrained ddpm
# implementation follows the algorithm by Ioan-Daniel Cracium, DDPM trained by Jiajae Fan
def bike_diffusion(parameter_csv_path: str, device: torch.device, ckpt_id: str='29000', mask_dilation: int=5, mask_fill_holes: bool=True, bike_idx: int=0, wheel_design_type: int=0, width: int=256, height: int=256):
    assert wheel_design_type == 0 or wheel_design_type == 1
    mask, background = get_mask_and_background(parameter_csv_path, bike_idx, wheel_design_type, width, height)
    bike_img = bike_inpainting(background, mask, device, 50, ckpt_id, mask_dilation, mask_fill_holes)
    return bike_img.convert('RGB')

# wrapper function for colorization using Stable Diffusion
# colorization is performed by first upscaling using Stable Diffusion and then performing masked img2img diffusion
def colorization(image: Image, colorization_model: str, upscaling_model: str, colorization_prompt: str, colorization_negative_prompt: str, fill_holes: bool, dilation: int, strength: float, prompt_guidance: float):
    colorized = sd_colorization(colorization_model, upscaling_model, image, colorization_prompt, negative_prompt=colorization_negative_prompt, fill_holes=fill_holes, dilation_iterations=dilation, colorization_strength=strength, prompt_guidance=prompt_guidance)
    return colorized

# wrapper function for the main part of our algorithm
def insert_diffusion(image: Image, mask_threshold: int, prompt: str, negative_prompt: str, img2img_model: str, inpainting_model: str, img2img_strength: float, inpainting_steps: int, inpainting_guidance: float, img2img_guidance: float, background_image: Image=None, composition_strength: float=1) -> Image:
    mask = get_mask_from_image(image, mask_threshold)
    
    # if we have a background image we paste the foreground onto the background by turning white pixels transparent
    if background_image is not None:
        image = paste_image(image, background_image)

    # perform masked inpainting
    inpainted = sd_inpainting(inpainting_model, image, mask, prompt, negative_prompt, inpainting_guidance, inpainting_steps, inpainting_strength=composition_strength)
    # perform rediffusion step
    result = sd_img2img(img2img_model, inpainted, prompt, negative_prompt, img2img_strength, img2img_guidance)
    return result

# wrapper function to extract an object with langSAM and paste it onto a white background of the same size as the original image
def extract_wrapper(image: Image, object_desc: str, background: Image=None) -> Image:
    if background is None:
        background = image
    # extract the object using langSAM and insert it onto a white background of the same size as the background image
    image = extract_object(image, background, object_desc)
    return image
    
if __name__ == '__main__':
    # -- CLI parameters --
    # detailed description of CLI parameters can be found in the readme
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--point_mode', action='store_true')
    parser.add_argument('--extract_object', type=str, default=None)

    parser.add_argument('--mask_threshold', type=int, default=175)
    parser.add_argument('--background_prompt', type=str, default=None)
    parser.add_argument('--negative_prompt', type=str, default="wrong proportions, toy, black frame, motorbike, silhouette, model, clay, high saddle, large wheels, text, above floor, flying, changed bike color, white background, duplicate, multiple, people, basket, distortion, low quality, worst, ugly, fuzzy, blurry, cartoon, simple, art")
    parser.add_argument('--auto_bike_prompt', action='store_true')
    parser.add_argument('--auto_car_prompt', action='store_true')
    parser.add_argument('--auto_product_prompt', action='store_true')
    parser.add_argument('--background_image', type=str, default=None)
    parser.add_argument('--composition_strength', type=float, default=1.)
    parser.add_argument('--img2img_model', type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0")
    parser.add_argument('--inpainting_model', type=str, default="stabilityai/stable-diffusion-2-inpainting")
    parser.add_argument('--img2img_strength', type=float, default=0.2)
    parser.add_argument('--inpainting_steps', type=int, default=75)
    parser.add_argument('--inpainting_guidance', type=float, default=15)
    parser.add_argument('--img2img_guidance', type=float, default=7.5)
    parser.add_argument('--output_folder', type=str, default='./images')

    parser.add_argument('--colorize', action='store_true')
    parser.add_argument('--colorization_model', type=str, default='diffusers/stable-diffusion-xl-1.0-inpainting-0.1')
    parser.add_argument('--upscaling_model', type=str, default='stabilityai/stable-diffusion-x4-upscaler')
    parser.add_argument('--colorization_prompt', type=str, default='A street bike with an orange frame a black seat, black tires, and black handles on a clean white background, 2D, colorful, side view')
    parser.add_argument('--colorization_negative_prompt', type=str, default='black and white, black frame, silhouette, motorbike, toy, clay, model, missing saddle, high saddle, details, detailed, greyscale, duplicate, multiple, detached, shadow, contact shadow, drop shadow, reflection, ground, unrealistic, bad, distorted, ugly, weird')
    parser.add_argument('--do_not_fill_holes', action='store_true')
    parser.add_argument('--dilation', type=int, default=8)
    parser.add_argument('--colorization_strength', type=float, default=0.91)
    parser.add_argument('--colorization_prompt_guidance', type=float, default=17)

    parser.add_argument('--datasheet_path', type=str, default=None)
    parser.add_argument('--bike_idx', type=int, default=0)
    parser.add_argument('--place', type=str, default=None)
    parser.add_argument('--color', type=str, default=None)

    parser.add_argument('--ckpt_id', type=str, default='290000')
    parser.add_argument('--bike_mask_dilation', type=int, default=5)
    parser.add_argument('--do_not_fill_bike_holes', action='store_true')
    parser.add_argument('--wheel_design', type=int, default=0)

    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--fraction_down', type=float, default=0.5)
    parser.add_argument('--fraction_right', type=float, default=0.5)
    parser.add_argument('--rotation', type=float, default=0)

    parser.add_argument('--car_manufacturer', type=str, default=None)
    parser.add_argument('--car_type', type=str, default=FileNotFoundError)
    parser.add_argument('--product_type', type=str, default=None)
    
    args = parser.parse_args()
    
    arg_image = args.image  # the image path if provided
    points = args.point_mode  # True if a bike image is to be generated from reference points before doing InsertDiffusion

    object_desc = args.extract_object  # None if object is already on white background

    # parameters relevant to generating a bike from reference points
    datasheet_path = args.datasheet_path
    ckpt_id = args.ckpt_id
    bike_mask_dilation = args.bike_mask_dilation
    bike_mask_fill_holes = not args.do_not_fill_bike_holes
    bike_idx = args.bike_idx
    wheel_design = args.wheel_design
    
    if arg_image is None and not points:
        raise ValueError('Algorithm needs either start image or reference points')
    if arg_image is not None and points:
        raise ValueError('Please provide either a start image or reference points, using both is not supported')
    # if point mode is used we employ the bike diffusion method to generate the starting image
    if points:
        if datasheet_path is None:
            raise ValueError('Please provide a datasheet path to run in point mode')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        image = bike_diffusion(datasheet_path, device, ckpt_id, bike_mask_dilation, bike_mask_fill_holes, bike_idx, wheel_design)
    # if an image path is provided the start image is simply loaded from disk
    elif arg_image is not None:
        image = Image.open(arg_image)
    
    background_image = args.background_image
    composition_strength = min(1, max(0, args.composition_strength))  # ensure 0-1 range
    # if a background image is provided and we are not using full composition strength we load the background image
    if background_image is not None and composition_strength < 1:
        background_image = Image.open(background_image)
    else:
        background_image is None
    # if no background image is provided we need to use full strength in inpainting
    if background_image is None:
        composition_strength = 1.
    
    # extract object from background using langSAM
    if object_desc is not None:
        image = extract_wrapper(image, object_desc, background_image)

    # requirements for using/generating prompt
    prompt = args.background_prompt
    auto_prompt_bike = args.auto_bike_prompt
    auto_prompt_car = args.auto_car_prompt
    auto_prompt_product = args.auto_product_prompt
    place = args.place
    if auto_prompt_bike or auto_prompt_car or auto_prompt_product:
        prompt = None
        if place is None:
            raise ValueError('You have to provide a place to use autoprompting')
    if int(auto_prompt_bike) + int(auto_prompt_car) + int(auto_prompt_product) > 1:
        raise ValueError('Can only generate one auto prompt')

    mask_threshold = args.mask_threshold
    negative_prompt = args.negative_prompt
    img2img_model = args.img2img_model
    inpainting_model = args.inpainting_model
    img2img_strength = args.img2img_strength
    inpainting_steps = args.inpainting_steps
    inpainting_guidance = args.inpainting_guidance
    img2img_guidance = args.img2img_guidance
    output_folder = args.output_folder
    colorize = args.colorize
    colorization_model = args.colorization_model
    upscaling_model = args.upscaling_model
    colorization_prompt = args.colorization_prompt
    colorization_negative_prompt = args.colorization_negative_prompt
    fill_holes = not args.do_not_fill_holes
    dilation = args.dilation
    colorization_strength = args.colorization_strength
    colorization_prompt_guidance = args.colorization_prompt_guidance
    color = args.color
    scale = args.scale
    fraction_down = args.fraction_down
    fraction_right = args.fraction_right
    rotation = args.rotation
    car_manufacturer = args.car_manufacturer
    car_type = args.car_type
    product_type = args.product_type

    if colorize and colorization_prompt is None and color is None:
        raise ValueError('You have specified to use colorization but have neither provided a colorization prompt nor a color')

    # generate prompts using predefined masks if no full prompt is provided
    if auto_prompt_bike:
        # bike prompts are generated using information from the dataset csv file
        bike_row = get_dataframe_row(datasheet_path, bike_idx)
        prompt = get_bike_inpainting_prompt(color, place, bike_row)
        if colorize and colorization_prompt is None:
            if not datasheet_path:
                raise ValueError('if you want to use auto-prompt generation for bikes please provide a datasheet path')
            bike_row = get_dataframe_row(datasheet_path, bike_idx)
            colorization_prompt = get_bike_colorization_prompt(color, bike_row)
    if auto_prompt_car:
        if car_manufacturer is None or car_type is None:
            raise ValueError('Please provide a color, car_manufacturer and car_type to use auto prompt generation for cars')
        prompt = get_car_inpainting_prompt(car_manufacturer, car_type, place, color if color is not None else '')
        if colorize and colorization_prompt is None:
            colorization_prompt = get_car_colorization_prompt(color, car_manufacturer, car_type)
    if auto_prompt_product:
        if product_type is None:
            raise ValueError('Please provide a product_type to use auto prompt generation for products')
        prompt = get_product_inpainting_prompt(product_type, place)
        if colorize and colorization_prompt is None:
            colorization_prompt = get_product_colorization_prompt(color, product_type)
    
    if colorize and colorization_prompt is None:
        raise ValueError('You specified to use colorization but have neither provided a colorization prompt nor are using auto prompting')

    print('\nRunning with prompts:')
    if colorize:
        print('Colorization:')
        print('\t'+colorization_prompt)
    print('Background: ')
    print('\t'+prompt)
    print()

    # scale and move object according to specification
    image = paste_pipeline(image, scale, fraction_down, fraction_right, rotation=rotation)
    
    # do image colorization if necessary
    if colorize:
        image = colorization(image, colorization_model, upscaling_model, colorization_prompt, colorization_negative_prompt, fill_holes, dilation, colorization_strength, colorization_prompt_guidance)

    # do InsertDiffusion
    result = insert_diffusion(image, mask_threshold, prompt, negative_prompt, img2img_model, inpainting_model, img2img_strength, inpainting_steps, inpainting_guidance, img2img_guidance, background_image, composition_strength)
    # save result to specified (or default) output folder while ensuring that the folder exists
    os.makedirs(output_folder, exist_ok=True)
    result.save(f'{output_folder}/{datetime.now().strftime("%d-%m-%Y--%H-%M-%S")}.png')
