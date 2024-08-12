# InsertDiffusion
This repository contains the official implementation for the paper "InsertDiffusion: Identity Preserving Visualization of Objects through a Training-Free Diffusion Architecture".

## Usage
The implementation supports two different running modes:

1. **image mode** - allows to "render" a preexisting image (arbitrary, as long as the background is white alternatively the object can be extracted from the original background by providing an object description) into an arbitrary background
2. **point mode** - *(not central to our research)* allows to create bike outlines from a point cloud and then insert the "rendered" bikes into an arbitrary background

In addition, the implementation offers the option to colorize the preexisting or generated images. When not using already colored images this drastically improves realism.

Further, a new environment can either be created from scratch (background replacement mode) or a preexisting image can be used and adapted.

All features are available from the `main.py` CLI and can also be integrated into your own projects. To use the CLI or the code an environment with torch, Pillow, pandas, transformers, and diffusers is required.

### Image mode
To insert an image into a newly created background, execute:
```bash
python main.py --image "<path to your image>" --background_prompt "<your prompt>"
```

So for example:
```bash
python main.py --image "./test_images/bikes/166.png" --background_prompt "a bicycle centered in frame in munich, 8k, red gemini, 50mm, f2.8, cinematic"
```

Instead of providing a prompt directly you can use the auto-prompting features by providing the necessary information. For bikes you need to provide the desired location, the index of the bike and a path to the datasheet. For cars you will have to provide the desired new location, the car manufacturer and car type. For products only the new location and a product type is required.

For further options and insertion into a given background consult the full CLI documentation below.

Instead of using the CLI, you may integrate the procedure into your code with this python function (again make sure to have all dependencies installed and copy the utils module to your project):
```python
from utils import get_mask_from_image, sd_inpainting, sd_img2img, paste_image

def insert_diffusion(image: Image, mask_threshold: int, prompt: str, negative_prompt: str, img2img_model: str, inpainting_model: str, img2img_strength: float, inpainting_steps: int, inpainting_guidance: float, img2img_guidance: float, background_image: Image=None, composition_strength: float=1) -> Image:
    mask = get_mask_from_image(image, mask_threshold)
    
    if background_image is not None:
        image = paste_image(image, background_image)

    inpainted = sd_inpainting(inpainting_model, image, mask, prompt, negative_prompt, inpainting_guidance, inpainting_steps, inpainting_strength=composition_strength)
    result = sd_img2img(img2img_model, inpainted, prompt, negative_prompt, img2img_strength, img2img_guidance)
    return result
```

### Point mode
We implemented an additional method to generate Biked images from point clouds and then insert them into a background. **This method does not work well, however, and is, therefore, *not central to our research*.** The generation from point clouds was implementation by *Ioan-Daniel Craciun* and is based upon a DDPM/DDIM implemented and trained by *Jiajae Fan*.

To create images from point clouds remove the `--image` argument from your CLI call. In point mode you have to provide a datasheet path via `--datasheet_path`. Thus, the minimal CLI command becomes:

```bash
python main.py --point --datasheet_path "<path to datasheet>" --background_prompt "<your prompt>"
```

You may also use auto-prompting in point mode.

To integrate the generation of bikes from point clouds into your project copy the utils folder and use the following function:
```python
from utils import get_mask_and_background
from utils import inpaint as bike_inpainting

def bike_diffusion(parameter_csv_path: str, device: torch.device, ckpt_id: str='29000', mask_dilation: int=5, mask_fill_holes: bool=True, bike_idx: int=0, wheel_design_type: int=0, width: int=256, height: int=256):
    assert wheel_design_type == 0 or wheel_design_type == 1
    mask, background = get_mask_and_background(parameter_csv_path, bike_idx, wheel_design_type, width, height)
    bike_img = bike_inpainting(background, mask, device, 50, ckpt_id, mask_dilation, mask_fill_holes)
    return bike_img.convert('RGB')
```

The image returned by this function is ready to be used for colorization or insertion.

### Colorization
If you are using an uncolored image add `--colorize` to one of the previous CLI calls. In addition, you need to provide a colorization prompt via `--colorization_prompt` or use auto-prompting by providing a color via `--color`.

An example CLI call could be:
```bash
python main.py --image "./test_images/bike_outline/168.png" --colorize  --datasheet_path "./csv/df_parameters_final.csv" --place "beach at sunset" --color "purple" --bike_idx 5"
```

Alternatively, to integrate colorization into your project, copy the utils module and use:
```python
from utils import sd_colorization

def colorization(image: Image, colorization_model: str, upscaling_model: str, colorization_prompt: str, colorization_negative_prompt: str, fill_holes: bool, dilation: int, strength: float, prompt_guidance: float):
    colorized = sd_colorization(colorization_model, upscaling_model, image, colorization_prompt, negative_prompt=colorization_negative_prompt, fill_holes=fill_holes, dilation_iterations=dilation, colorization_strength=strength, prompt_guidance=prompt_guidance)
    return colorized
```

### CLI arguments
*Note*: CLI is still under construction and might be subject to change.

*Note*: Where applicable default values represent the parameters determined in our experiments. But different values might be optimal depending on the specific use case.

| argument | type | description |
| -- | -- | -- |
| --image | string | path to a image to start from, mutually exclusive with --points |
| --points | | when points is used the algorithm runs in point cloud mode and generates a bike outline from a point cloud first, mutually exclusive with --image |
| --mask_threshold | int | for inpainting, threshold to discern white background from colored foreground |
| --background_prompt | string | the prompt for the background generation |
| --negative_prompt | string | negative prompt for background generation |
| --background_image | string | string to a background image to be used as a starting point, only relevant if --composition strength is set to a value smaller than 1 |
| --composition_strength | float | determines how much to change the starting point background image, only used is --background_image is set, range 0-1 |
| --auto_bike_prompt | | if set prompts will be automatically created using the template for bikes, requires --place, --datasheet_path, and --bike_idx to be set |
| --auto_car_prompt |  | if set prompts will be automatically created using the template for cars, requires --place, --car_manufacturer, and --car_type to be set |
| --auto_product |  | if set prompts will be automatically created using the template for products, requires --place, and --product_type to be set |
| --place | string | description of the location where the object is to be inserted, only used if one of the auto prompt templates is used |
| --color | string | if using autoprompting, which color does the bike have |
| --datasheet_path | string | if using autoprompting for bikes, path to datasheet for lookup of bike type |
| --bike_idx | int | if using autoprompting for bikes, index in datasheet for lookup of bike type |
| --car_manufacturer | string | if using autoprompting for cars, manufacturer of the car e.g. BMW |
| --car_type | string | if using autoprompting for cars, type of the car e.g. SUV or X5 |
| --product_type | string | if using autoprompting for products, type of the product e.g. lamp |
| --inpainting_model | string | which model to use for background generation (huggingface id) |
| --img2img_model | string | which model to use for rediffusion step (huggingface id) |
| --img2img_strength | float | how much of the original image to noise in rediffusion |
| --inpainting_steps | int | how many diffusion steps to perform for inpainting |
| --inpainting_guidance | float | how much classifier-free guidance to apply in inpainting |
| --img2img_guidance | float | how much classifier-free guidance to apply in rediffusion |
| --output_folder | string | path to folder in which output images should be saved |
| --colorize | | whether to colorize image before inpainting |
| --colorization_model | string | which model to use for colorization (huggingface id) |
| --upscaling_model | string | which model to use for upscaling, necessary for colorization (huggingface id) |
| --colorization_prompt | string | prompt for colorization, not necessary is datasheet_path and color are provided|
| --colorization_negative_prompt | string | negative prompt for colorization |
| --do_not_fill_holes | | toggle hole filling for colorization mask, only relevant when colorizing |
| --dilation | int | how much to extend the mask for colorization, only relevant when colorizing |
| --colorization_strength | float | how much diffusion to apply for colorization, only relevant when colorizing |
| --colorization_prompt_guidance | float | how much classifier-free guidance to apply during colorization, only relevant when colorizing |
| --scale | float | how much to down or upscale the bike, higher values result in the bike taking up more space in frame, default is 1. |
| --fraction_down | float | relative y position of the (center of the) bike, higher values place the bike closer to the lower edge of the image, default is 0.5 (centered) |
| --fraction_right | float | relative x position of the (center of the) bike, higher values place the bike closer to the right edge of the image, default is 0.5 (centered) |
| --ckpt_id | string | id of the checkpoint to use for the creation of bike outlines from point clouds, only relevant in point mode |
| --bike_mask_dilation | int | how much to extend the masks generated from the point clouds, only relevant in point mode |
| --do_not_fill_bike_holes | | whether to apply hole filling to bike masks, only relevant in point mode |
| --wheel_design | int | which wheel design to use for the generation of bike outlines, currently only 0 and 1 are implemented, only relevant in point mode |

### Interactive Generation
An additional script `interactive.py` provides an implementation to generate images interactively in a Human-in-the-Loop fashion. This means, at each step five options are generated in parallel and the user is asked to choose the best option. Then, only the chosen image is used for the next step.

No additional CLI arguments are accepted by the script the user will be prompted to make all decisions.

To run in interactive mode execute:

```bash
python interactive.py
```

## Evaluation
The evaluation code used or our paper can be found under `./evaluation`. Note that quantitative metrics were computed using the following CLI command:
```bash
python evaluate.py --exp_name "<experiment name>" --gen_file_path "<path to generated images>" --ref_file_path "<path to reference files>" --masks_path "<only used for composition, path to masks>"
```
Human-Evaluation metrics and inferential statistics were computed using the provided notebook.

## Citation
If find our work helpful and want to use it for your research or project, please cite the paper as follows:

```bibtex
@misc{2407.10592,
Author = {Phillip Mueller and Jannik Wiese and Ioan Craciun and Lars Mikelsons},
Title = {InsertDiffusion: Identity Preserving Visualization of Objects through a Training-Free Diffusion Architecture},
Year = {2024},
Eprint = {arXiv:2407.10592},
}
```
