import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import os
import sys
from scipy.ndimage import binary_erosion

parent_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the system path
sys.path.append(parent_dir)

from lang_sam import LangSAM
import torch

# in this file we use the lang_sam package by Luca Medeiros (https://github.com/luca-medeiros/lang-segment-anything) to extract objects from the original background and insert them into a clean white background

# model as a global variable s.t. it is not loaded new everytime the functions in the file are used
model = None

# this method is used to generate segmentation masks and bounding boxes using a prompt
def get_mask_and_bb_langsam(image: Image, prompt: str):
    global model
    if model is None:
        model = LangSAM()
    # use langSAM to predict all bounding boxes and segmentation masks for a given prompt
    masks, boxes, phrases, logits = model.predict(image, prompt)

    # if no object is found return None None
    if len(masks) == 0:
        return None, None
    # if only a single mask is found the mask is returned
    if len(masks) == 1:
        return masks.detach().squeeze().cpu().numpy(), boxes.detach().squeeze().cpu().numpy()

    # old code to combine all found masks into a single mask:
    # combined_mask = torch.any(masks, dim=0)
    # return combined_mask.detach().cpu().numpy(), boxes[0].detach().cpu().numpy()

    # if more than one mask is found the best fitting mask is returned (maximal logit indicates highest probability)
    max_logit_idx = torch.argmax(logits).item()
    return masks[max_logit_idx].detach().cpu().numpy(), boxes[max_logit_idx].detach().cpu().numpy()

# this function extracts an object from an image based on the objects description (prompt) and pastes it onto a white image
def get_pasted_image(img: Image, prompt: str, erosion_strength: int):
    # get segmentation mask of object
    mask, _ = get_mask_and_bb_langsam(img, prompt)
    if mask is None:
        # if no fitting object is found an empty mask is created
        print('Object not found!')
        mask = np.zeros_like(np.array(img)).astype(bool)
    if erosion_strength > 0:
        # if a mask is found it is eroded to produce a cleaner mask
        mask = binary_erosion(mask, iterations=erosion_strength)
    im_array = np.array(img)
    # create a pure black image
    im_pasted = np.ones_like(im_array)*255
    # replace masked values to insert object onto background at original position
    im_pasted[mask] = im_array[mask]
    return Image.fromarray(im_pasted)
    