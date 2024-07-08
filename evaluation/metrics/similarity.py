import torch
import torchvision.transforms as T
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from PIL import Image
import os
import sys

# DEBUG
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the system path
sys.path.append(parent_dir)

from lang_sam import LangSAM

langsam = None

def img_list_to_tensors(imgs: list, normalize: bool=True):
    if normalize:
        transform = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        transform = T.Compose([
            T.PILToTensor()
        ])

    imgs = [transform(x) for x in imgs]
    # imgs = torch.stack([transform(x) for x in imgs])
    # assert imgs.size(1) == 3
    return imgs

def get_mask_and_bb_langsam(image: Image, prompt: str):
    global langsam
    if langsam is None:
        langsam = LangSAM()
    masks, boxes, phrases, logits = langsam.predict(image, prompt)

    # print(f'found {len(masks)} objects')

    if len(masks) == 0:
        return None, None

    if len(masks) == 1:
        return masks.detach().squeeze().cpu().numpy(), boxes.detach().squeeze().cpu().numpy()

    max_logit_idx = torch.argmax(logits).item()
    return masks[max_logit_idx].detach().cpu().numpy(), boxes[max_logit_idx].detach().cpu().numpy()

def get_inserted_bouding_box_sam(imgs: list, objects: list) -> list:
    global langsam
    if langsam is None:
        langsam = LangSAM()

    bounding_boxes = []
    for i, img in enumerate(imgs):
        object_desc = objects[i]
        _, bb = get_mask_and_bb_langsam(img, object_desc)
        if bb is None:
            bb = (0, 0, img.width, img.height)
        bounding_boxes.append(bb)
    return bounding_boxes

def get_inserted_bounding_box_mask(masks: list) -> list:
    bbs = []
    for mask in masks: 
        width, height = mask.size
        leftmost = width
        topmost = height
        rightmost = 0
        bottommost = 0

        for x in range(width):
            for y in range(height):
                if mask.getpixel((x, y)) == (255, 255, 255):
                    leftmost = min(leftmost, x)
                    topmost = min(topmost, y)
                    rightmost = max(rightmost, x)
                    bottommost = max(bottommost, y)
        
        if rightmost < leftmost or bottommost < topmost:
            bbs.append((0, 0, width, height))
        else: 
            bbs.append((leftmost, topmost, rightmost-leftmost, bottommost-topmost))
        
    return bbs

def lpips(gen_imgs: list, original_images: list, bbs: list) -> float:
    for i, bb in enumerate(bbs):
        bb = list(bb)
        if bb[2] < bb[0]:
            bb[2] = bb[0] + 17
            bb[0] = bb[0] - 16
        if bb[3] < bb[1]:
            bb[3] = bb[1] + 17
            bb[1] = bb[1] - 16
        bbs[i] = tuple(bb)
    
    for i, bb in enumerate(bbs):
        bb = list(bb)
        if bb[2] - bb[0] < 20:
            bb[2] = bb[2] + (20 - (bb[2] - bb[2]))
        if bb[3] - bb[1] < 20:
            bb[3] = bb[3] + (20 - (bb[3] - bb[1]))
        bbs[i] = tuple(bb)

    patch_widths = [x[2]-x[0] for x in bbs]
    patch_heights = [x[3]-x[1] for x in bbs]

    original_images = [x.resize((patch_widths[i], patch_heights[i]), Image.Resampling.BICUBIC) for i, x in enumerate(original_images)]
    gen_imgs = [x.crop((bbs[i][0], bbs[i][1], bbs[i][2], bbs[i][3])).resize((patch_widths[i], patch_heights[i]), Image.Resampling.NEAREST) for i, x in enumerate(gen_imgs)]

    gen_imgs = img_list_to_tensors(gen_imgs)
    original_images = img_list_to_tensors(original_images)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    lpipss = [lpips(original_images[i].unsqueeze(0), gen_imgs[i].unsqueeze(0)).detach().item() for i in range(len(gen_imgs))]

    return sum(lpipss)/len(lpipss)

def ssim(gen_imgs: list, original_images: list, bbs: list) -> float:
    for i, bb in enumerate(bbs):
        bb = list(bb)
        if bb[2] < bb[0]:
            bb[2] = bb[0] + 17
            bb[0] = bb[0] - 16
        if bb[3] < bb[1]:
            bb[3] = bb[1] + 17
            bb[1] = bb[1] - 16
        bbs[i] = tuple(bb)

    for i, bb in enumerate(bbs):
        bb = list(bb)
        if bb[2] - bb[0] < 20:
            bb[2] = bb[2] + (20 - (bb[2] - bb[2]))
        if bb[3] - bb[1] < 20:
            bb[3] = bb[3] + (20 - (bb[3] - bb[1]))
        bbs[i] = tuple(bb)

    patch_widths = [x[2]-x[0] for x in bbs]
    patch_heights = [x[3]-x[1] for x in bbs]

    original_images = [x.resize((patch_widths[i], patch_heights[i]), Image.Resampling.BICUBIC) for i, x in enumerate(original_images)]
    gen_imgs = [x.crop((bbs[i][0], bbs[i][1], bbs[i][2], bbs[i][3])).resize((patch_widths[i], patch_heights[i]), Image.Resampling.NEAREST) for i, x in enumerate(gen_imgs)]
    
    gen_imgs = img_list_to_tensors(gen_imgs)
    original_images = img_list_to_tensors(original_images)

    ssim = StructuralSimilarityIndexMeasure()
    ssims = [ssim(gen_imgs[i].unsqueeze(0), original_images[i].unsqueeze(0)).detach().item() for i in range(len(gen_imgs))]

    return sum(ssims)/len(ssims)
