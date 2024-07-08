from PIL import Image
import os
from .mask import get_mask_from_image

def get_bikes_and_masks(bike_image_path: str='./test_images/bikes/', threshold: int = 175) -> tuple:
    bike_paths = os.listdir(bike_image_path)
    bike_paths = [bike_image_path + x for x in bike_paths]
    bikes = [Image.open(x) for x in bike_paths]
    masks = [get_mask_from_image(x, threshold) for x in bikes]
    return bikes, masks