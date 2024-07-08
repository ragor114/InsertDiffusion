import pandas as pd

# -- prompt templates for bikes, cars and products --

BIKE_COLORIZATION_PROMPT_TEMPLATE = "A <bike_type> bike with a <color> frame a black seat, black tires, and black handles on a clean white background, bicycle, 2D, colorful, side view"
BIKE_INPAINTING_PROMPT_TEMPLATE = "A <color> <bike_type> bike on a <place>, bicycle, sony a7sii, 35mm, 8k, cinematic, high quality"

CAR_COLORIZATION_PROMPT_TEMPLATE = "A <color> <car_manufacturer> <car_type> with black wheels on a clean white background, car, 2D, colorful, side view"
CAR_INPAINTING_PROMPT_TEMPLATE = "A <color> <car_manufacturer> <car_type> on a <place>, car, sony a7sii, 35mm, 8k, cinematic, high quality"

PRODUCT_COLORIZATION_PROMPT_TEMPLATE = "A <color> <product_type> on a clean white background, bicycle, 2D, colorful, side view"
PRODUCT_INPAINTING_PROMPT_TEMPLATE = "A <product_type> on a <place>, product, advertisement, sony a7sii, 35mm, 8k, cinematic, high quality"

# -- bike prompt --

# extract bike type from Dataframe row
def get_bike_type(bike_row: pd.Series) -> str:
    bike_type = bike_row['BikeStyle']
    bike_type = bike_type.lower()
    bike_type = bike_type.replace('_', ' ')

    if bike_type == 'mtb':
        bike_type = 'mountain'
    if bike_type == 'other':
        bike_type = ''
    
    return bike_type

def get_bike_colorization_prompt(color: str, bike_row: pd.Series) -> str:
    bike_type = get_bike_type(bike_row)
    prompt = BIKE_COLORIZATION_PROMPT_TEMPLATE.replace('<bike_type>', bike_type).replace('<color>', color)
    return prompt

def get_bike_inpainting_prompt(color: str, place: str, bike_row: pd.Series) -> str:
    bike_type = get_bike_type(bike_row)
    prompt = BIKE_INPAINTING_PROMPT_TEMPLATE.replace('<color>', color).replace('<bike_type>', bike_type).replace('<place>', place)
    return prompt

def get_bike_prompts(color: str, place: str, bike_row: pd.Series) -> str:
    return get_bike_colorization_prompt(color, bike_row), get_bike_inpainting_prompt(color, place, bike_row)

# -- car prompt --

def get_car_colorization_prompt(color: str, car_manufacturer: str, car_type: str) -> str:
    prompt = CAR_COLORIZATION_PROMPT_TEMPLATE.replace('<car_type>', car_type).replace('<color>', color).replace('<car_manufacturer>', car_manufacturer)
    return prompt

def get_car_inpainting_prompt(car_manufacturer: str, car_type: str, place: str, color: str='') -> str:
    prompt = CAR_INPAINTING_PROMPT_TEMPLATE.replace('<color>', color).replace('<car_type>', car_type).replace('<car_manufacturer>', car_manufacturer).replace('<place>', place)
    return prompt

def get_car_prompts(car_manufacturer: str, car_type: str, place: str, color: str='') -> str:
    return get_car_colorization_prompt(color, car_manufacturer, car_type), get_car_inpainting_prompt(car_manufacturer, car_type, place, color)

# -- product prompt --

def get_product_colorization_prompt(color: str, product_type: pd.Series) -> str:
    prompt = PRODUCT_COLORIZATION_PROMPT_TEMPLATE.replace('<product_type>', product_type).replace('<color>', color)
    return prompt

def get_product_inpainting_prompt(product_type: str, place: str) -> str:
    prompt = PRODUCT_INPAINTING_PROMPT_TEMPLATE.replace('<product_type>', product_type).replace('<place>', place)
    return prompt

def get_product_prompts(product_type: str, color: str, place: str) -> str:
    return get_product_colorization_prompt(color, product_type), get_product_inpainting_prompt(product_type, place)