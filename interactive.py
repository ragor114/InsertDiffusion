import matplotlib
matplotlib.use('TkAgg')  # necessary for plots to show up as GUI

from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import torch

from utils import get_mask_and_background, get_mask_from_image, sd_inpainting, sd_img2img, sd_colorization, get_dataframe_row, extract_object, paste_pipeline, paste_image
from utils import get_bike_colorization_prompt, get_bike_inpainting_prompt, get_car_colorization_prompt, get_car_inpainting_prompt, get_product_colorization_prompt, get_product_inpainting_prompt
from utils import inpaint as bike_inpainting

NUM_IMAGES = 5

COLORIZATION_MODEL = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
UPSCALING_MODEL = 'stabilityai/stable-diffusion-x4-upscaler'
COLORIZATION_NEGATIVE_PROMPT = 'black and white, black frame, silhouette, motorbike, toy, clay, model, missing saddle, high saddle, details, detailed, greyscale, duplicate, multiple, detached, shadow, contact shadow, drop shadow, reflection, ground, unrealistic, bad, distorted, ugly, weird'
COLORIZATION_FILL_HOLES = True
COLORIZATION_DILATION = 8
COLORIZATION_STRENGTH = 0.91
COLORIZATION_GUIDANCE = 17
COLORIZATION_MASK_THRESHOLD = 170
COLORIZATION_STEPS = 30

INPAINTING_MASK_THRESHOLD = 175
INPAINTING_MODEL = 'stabilityai/stable-diffusion-2-inpainting'
INPAINTING_NEGATIVE_PROMPT = 'wrong proportions, toy, black frame, motorbike, silhouette, model, clay, high saddle, large wheels, text, above floor, flying, changed bike color, white background, duplicate, multiple, people, basket, distortion, low quality, worst, ugly, fuzzy, blurry, cartoon, simple, art'
INPAINTING_GUIDANCE = 15
INPAINTING_STEPS = 75

IMG2IMG_MODEL = 'stabilityai/stable-diffusion-xl-refiner-1.0'
IMG2IMG_STRENGTH = 0.2
IMG2IMG_GUIDANCE = 7.5

GENERATION_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
GENERATION_STEPS = 50
GENERATION_DILATION = 7
GENERATION_CKPT = '290000'
GENERATION_FILL_HOLES = True

# idea, generate a batch of 5 images in each step and let a human decide for the best image
# TODO: for now only auto-prompting

# The user first selects whether to start from a predefined image or generate a bike image from a point cloud
def read_image_mode_selected():
    selection = input('Do you want to start with an image [i] or from a point cloud [p]? ')
    if selection != 'p':  # default: use image mode
        return True
    return False

# in image mode the user first has to provide the path to the start image
def read_image_path():
    path = input('Enter the path to the initial image (e.g. ./test_images/bike_outline/166.png): ')
    if path == '':
        return './test_images/bike_outline/166.png'
    return path

# to generate a prompt based on the predefined masks the user has to describe the environment in which the object is to be inserted
def read_place():
    place = input('Where would you like to see the object (e.g. downtown manhattan street at sunset): ')
    if place == '':
        place = 'downtown manhattan street at sunset'
    return place

# if colorization is supposed to be used the user has to declare the desired color
def read_color():
    color = input('Enter a color if you would like to use colorization: ')
    return color

# to use auto-prompting for bikes and to generate bikes from reference points the user has to provide the path to the csv file as well as the index of the bike they are using
def read_datasheet_path():
    datasheet_path = input('Enter the path to the datasheet (e.g. ./csv/df_parameters_final.csv): ')
    if datasheet_path == '':
        datasheet_path = './csv/df_parameters_final.csv'
    bike_idx = input('Which row number in the datasheet refers to your bike (e.g. 1): ')
    if bike_idx == '':
        return datasheet_path, 1
    bike_idx = int(bike_idx)
    return datasheet_path, bike_idx

# for generating a bike from reference points, wheels are used as an additional "conditioning"
# TODO: currently only two different wheel designs are provided
def read_wheel_type():
    wheel_type = input('Which wheel type should be used [0] - BMX/MTB, [1] - Street/Racing: ')
    if wheel_type == '':
        return 0
    return int(wheel_type)

# this function is used to generate a predefined number of bikes from reference points
def batched_generation_from_points(datasheet_path: str, bike_idx: int, wheel_type: int) -> list:
    # get the mask for restricted generation based on reference points as well as a white background with the wheels
    mask, background = get_mask_and_background(datasheet_path, bike_idx, wheel_type, 256, 256)
    # use the pretrained DDPM to generate bikes
    bikes = bike_inpainting(background, mask, GENERATION_DEVICE, GENERATION_STEPS, GENERATION_CKPT, GENERATION_DILATION, GENERATION_FILL_HOLES, NUM_IMAGES)
    return bikes

# this function is used to colorize the same image multiple times
def batched_colorization(image: Image, colorization_prompt: str) -> list:
    colorized_imgs = sd_colorization(COLORIZATION_MODEL, UPSCALING_MODEL, image, colorization_prompt, COLORIZATION_NEGATIVE_PROMPT, COLORIZATION_MASK_THRESHOLD, COLORIZATION_FILL_HOLES, COLORIZATION_DILATION, COLORIZATION_GUIDANCE, COLORIZATION_STEPS, COLORIZATION_STRENGTH, NUM_IMAGES)
    return colorized_imgs

# this funciton is used to perform multiple different masked inpaintings
def batched_inpainting(image: Image, inpainting_prompt: str, background_image: Image, composition_strength) -> list:
    mask = get_mask_from_image(image, INPAINTING_MASK_THRESHOLD)
    if background_image is not None and composition_strength < 1:
        image = paste_image(image, background_image)
    inpainted = sd_inpainting(INPAINTING_MODEL, image, mask, inpainting_prompt, INPAINTING_NEGATIVE_PROMPT, INPAINTING_GUIDANCE, INPAINTING_STEPS, NUM_IMAGES, inpainting_strength=composition_strength)
    return inpainted

# finally, this funciton is used to generate multiple different results of rediffusion
def batched_insertion(image: Image, inpainting_prompt: str) -> list:
    result = sd_img2img(IMG2IMG_MODEL, image, inpainting_prompt, INPAINTING_NEGATIVE_PROMPT, IMG2IMG_STRENGTH, IMG2IMG_GUIDANCE, NUM_IMAGES)
    return result

# this function is used to display the options generated in the previous step and let the user decide for the image used to procede
def selection(imgs: list, step: str) -> Image:
    print('Close the image viewer when you have decided!')

    # plotting of all options
    title = f'Which {step} is best?'
    plt.figure(figsize=(12, 6))
    plt.suptitle(title)
    for i, im in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.title(f'{i+1}')
        plt.axis('off')
        plt.imshow(im)
    plt.tight_layout()
    plt.show()

    print()
    # selection by user
    selected = input(f'Which {step} is best: ')
    if selected == '':
        return imgs[0]
    selected = int(selected) - 1  # [1, num_images] to [0, num_images-1]
    return imgs[selected]

# auto prompt generation
def get_prompts(bike_row=None):
    colorization_prompt = None  # by default no colorization prompt is used

    # user inputs either full prompt or which autoprompting technique is to be used
    prompt_type = input("Enter a full inpainting prompt or enter 'bike', 'car' or 'product' to use one of the autoprompting techniques: ")
    prompt_type = prompt_type.strip()
    if prompt_type == '':
        prompt_type = 'bike'

    # auto prompting for bikes is based on data in csv file, also requires place and can optionally colorize
    if prompt_type == 'bike':
        place = read_place()
        if bike_row is None:
            datasheet_path, bike_idx = read_datasheet_path()
            bike_row = get_dataframe_row(datasheet_path, bike_idx)
        color = read_color().strip()
        if len(color) > 0:
            colorization_prompt = get_bike_colorization_prompt(color, bike_row)
        inpainting_prompt = get_bike_inpainting_prompt(color, place, bike_row)
    # auto prompting for car rquires place, color, car_type and car_manufacturer
    elif prompt_type == 'car':
        place = read_place()
        color = read_color()
        car_type = input('Enter the type of car (e.g. SUV or X5): ')
        car_type = car_type.strip()
        car_type = 'car' if car_type == '' else car_type
        car_manufacturer = input('Enter the car manufacturer (e.g. BMW): ')
        car_manufacturer = car_manufacturer.strip()
        inpainting_prompt = get_car_inpainting_prompt(car_manufacturer, car_type, place, color)
    # auto prompting for products requires only the type of product and a location
    elif prompt_type == 'product':
        place = read_place()
        product_type = input('Enter the product type (e.g. lamp): ')
        product_type = product_type.strip()
        product_type = 'object' if product_type == '' else product_type
        inpainting_prompt = get_product_inpainting_prompt(product_type, place)
    # the user might also provide a full prompt directly
    else:
        inpainting_prompt = prompt_type

    return colorization_prompt, inpainting_prompt

# this function wraps colorization, inpainting and rediffusion in interactive mode
def interactive_insert_diffusion(image: Image, background_image: Image, composition_strength: float):
    # get prompts
    colorization_prompt, inpainting_prompt = get_prompts()

    # colorization (optional)
    if colorization_prompt is not None:
        colorized_imgs = batched_colorization(image, colorization_prompt)
        colorized_image = selection(colorized_imgs, 'colorization')
    else:
        colorized_image = image

    # masked inpainting
    inpainted_imgs = batched_inpainting(colorized_image, inpainting_prompt, background_image, composition_strength)
    inpainted_img = selection(inpainted_imgs, 'inpainting')

    # rediffusion
    rediffused_imgs = batched_insertion(inpainted_img, inpainting_prompt)
    final_img = selection(rediffused_imgs, 'insertion')
    return final_img

# this function is used to let the user select a reference image, and optionally a background image
def get_images():
    # let the user provide a path to the reference image
    image_path = read_image_path()
    image = Image.open(image_path)

    # if the object has to be extracted from its original background first, a description of the object is needed for langSAM
    object_desc = input('If you want to extract an object from it`s original background enter an object description: ')

    # the user can optionally provide a background image in which the object shall be inserted
    background_image_path = input('If you want to insert the object into an existing background enter the background image path: ')
    if background_image_path is not None and background_image_path != '':
        background_image = Image.open(background_image_path)

        # if the user provides a background image they have to decide how much of the original image to retain (0) or remove (1)
        composition_strength = input('How much do you want to change the background image from 0 (no change) to 1 (full replacement)? ')
        composition_strength = 1. if composition_strength == '' else max(0., min(1., float(composition_strength)))
    else:
        background_image, composition_strength = None, 1.

    if object_desc is not None and object_desc != '':
        if background_image is None:
            bg = image
        else:
            bg = background_image
        # use langSAM to extract object from background and paste it onto a white background of the same size as the background (or original image if not provided)
        image = extract_object(image, bg, object_desc)
    
    # optionally the user can reposition the object in frame by providing scale, down and right positionings
    scale = input('How would you like to scale the object (0-100, 1 = no change)? ')
    scale = 1. if scale == '' else float(scale)
    fraction_down = input('Would you like to move the object down (0-1, 0.5 = no change)? ')
    fraction_down = 0.5 if fraction_down == '' else float(fraction_down)
    fraction_right = input('Would you like to move the object horizontally (0-1, 0.5 = no change)? ')
    fraction_right = 0.5 if fraction_right == '' else float(fraction_right)
    image = paste_pipeline(image, scale, fraction_down, fraction_right)  # move object based on provided values

    return image, background_image, composition_strength

# interactive image mode generation
def image_mode_generation():
    print()
    image, background_image, composition_strength = get_images()
    final_img = interactive_insert_diffusion(image, background_image, composition_strength)
    return final_img

# interactive generation based on reference points
# Currently interactive point mode generation is only supported for background replacement
def point_mode_generation():
    datasheet_path, bike_idx = read_datasheet_path()
    wheel_type = read_wheel_type()
    bike_row = get_dataframe_row(datasheet_path, bike_idx)
    generated = batched_generation_from_points(datasheet_path, bike_idx, wheel_type)
    image = selection(generated, 'shape generation')
    final_img = interactive_insert_diffusion(image, bike_row, 1.)
    return final_img

# main function
def interactive_generation():
    # user selects mode and proceeds with the appropriate function
    image_mode_selected = read_image_mode_selected()
    if image_mode_selected:
        image = image_mode_generation()
    else:
        image = point_mode_generation()

    exp_time = datetime.now()
    image.save(f'./images/interactive_{exp_time.strftime("%d-%m-%Y--%H-%M-%S")}.png')

if __name__ == '__main__':
    interactive_generation()