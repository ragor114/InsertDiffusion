import os
from PIL import Image
import argparse
import pandas as pd
import metrics
from datetime import datetime
import numpy as np
import sys
sys.path.insert(1, '../InsertDiffusion')
import utils

#DEBUG
from torchmetrics.functional.multimodal import clip_score
import torchvision.transforms as T
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def load_masks(path: str) -> list:
    file_names = os.listdir(path)
    file_names.sort()
    mask_imgs = [Image.open(os.path.join(path, x)) for x in file_names]
    return mask_imgs

def load_gen_files(fp: str) -> tuple:
    img_files = os.listdir(fp)
    img_files.sort()
    print('img files')
    print(img_files)
    prompts = [x.split('.')[0] for x in img_files]
    for i, prompt in enumerate(prompts):
        while prompt[0].isnumeric():
            prompt = prompt[1:]
        prompts[i] = prompt
    imgs = [Image.open(os.path.join(fp, x)) for x in img_files]
    return imgs, prompts

def extract_object(fg: Image, bg: Image, object_desc: str) -> Image:
    if fg.height > fg.width:
        factor = bg.height / fg.height
        fg = fg.resize((int(fg.width*factor), int(fg.height*factor)), Image.Resampling.NEAREST)
    else:
        factor = bg.width / fg.width
        fg = fg.resize((int(fg.width*factor), int(fg.height*factor)), Image.Resampling.NEAREST)

    fg = utils.get_pasted_image(fg, object_desc, erosion_strength=0)
    fg = utils.paste_pipeline(fg, 1, 0.5, 0.5)
    img_white = Image.new('RGB', (bg.width, bg.height), color='white')
    img_white.paste(fg, (img_white.width//2-fg.width//2, img_white.height//2-fg.height//2))

    fg_arr = np.array(img_white)
    non_white = np.where((fg_arr < 200).any(axis=2))

    bb_left, bb_right, bb_top, bb_bottom = max(np.min(non_white[1])-2, 0), min(np.max(non_white[1])+2, fg_arr.shape[1]), max(np.min(non_white[0])-2, 0), min(np.max(non_white[0])+2, fg_arr.shape[0])
    extracted = img_white.crop((bb_left, bb_top, bb_right, bb_bottom))

    return extracted

def load_ref_files(fp: str, bgs: list) -> tuple:
    img_files = os.listdir(fp)
    img_files.sort()
    print('ref files')
    print(img_files)
    imgs = [Image.open(os.path.join(fp, x)) for x in img_files]
    object_descs = [x.split('.')[0].split('_')[-1] for x in img_files]

    # crop ref images to only show the object (set all other pixels white)
    ref_imgs = []
    for i, x in enumerate(imgs):
        try:
            ref_img = extract_object(x, bgs[i], object_descs[i])
        except ValueError:
            # TODO: remove later - works for TFI-benchmark dataset but probably not for anything else
            img_white = Image.new('RGB', (bgs[i].width, bgs[i].height), color='white')
            img_white.paste(x, (img_white.width//2-x.width//2, img_white.height//2-x.height//2))

            fg_arr = np.array(img_white)
            non_white = np.where((fg_arr < 200).any(axis=2))

            bb_left, bb_right, bb_top, bb_bottom = max(np.min(non_white[1])-2, 0), min(np.max(non_white[1])+2, fg_arr.shape[1]), max(np.min(non_white[0])-2, 0), min(np.max(non_white[0])+2, fg_arr.shape[0])
            ref_img = img_white.crop((bb_left, bb_top, bb_right, bb_bottom))
        ref_imgs.append(ref_img)
    # imgs = [extract_object(x, bgs[i], object_descs[i]) for i, x in enumerate(imgs)]
    imgs = ref_imgs

    return imgs, object_descs

def load_gt_files(fp: str) -> tuple:
    img_files = os.listdir(fp)
    img_files.sort()
    imgs = [Image.open(os.path.join(fp, x)) for x in img_files]
    return imgs

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='experiment')

    parser.add_argument('--gen_file_path', type=str, required=True)
    parser.add_argument('--ref_file_path', type=str, default=None)
    parser.add_argument('--gt_file_path', type=str, default=None)
    parser.add_argument('--masks_path', type=str, default=None)

    parser.add_argument('--no_is', action='store_true')
    parser.add_argument('--no_fid', action='store_true')
    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--no_lpips', action='store_true')
    parser.add_argument('--no_ssim', action='store_true')
    parser.add_argument('--no_hps', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    gen_imgs, prompts = load_gen_files(args.gen_file_path)

    ref_imgs, object_descs = None, None
    gt_imgs = None
    mask_imgs = None

    if args.ref_file_path:
        ref_imgs, object_descs = load_ref_files(args.ref_file_path, gen_imgs)
    if args.gt_file_path:
        gt_imgs = load_gt_files(args.gt_file_path)
    if args.masks_path:
        # mask images are used for compositing because we found that objects might be distorted too much to be found by SAM
        mask_imgs = load_masks(args.masks_path)
    
    if args.no_is:
        inc_score = -1
    else:
        inc_score = metrics.inception_score(gen_imgs)

    if args.gt_file_path is None or args.no_fid:
        fid = -1
    else:
        fid = metrics.frechet_inception_distance(gen_imgs, gt_imgs)
    
    if args.no_clip:
        clip_score = -1
    else:
        clip_score = metrics.clip_score(gen_imgs, prompts)

    if args.ref_file_path is None:
        lpips = -1
        ssim = -1
    else:
        if mask_imgs:
            bbs = metrics.get_inserted_bounding_box_mask(mask_imgs)
        else:
            bbs = metrics.get_inserted_bouding_box_sam(gen_imgs, object_descs)

        if args.no_lpips:
            lpips = -1
        else:
            lpips = metrics.lpips(gen_imgs, ref_imgs, bbs)
        if args.no_ssim:
            ssim = -1
        else:
            ssim = metrics.ssim(gen_imgs, ref_imgs, bbs)
    
    if args.no_hps:
        hps = -1
    else:
        hps = metrics.hps(gen_imgs, prompts)

    if not os.path.exists('./results/results.csv'):
        os.makedirs('./results', exist_ok=True)
        df = pd.DataFrame(columns=['Timestamp', 'Experiment', 'IS', 'FID', 'CLIP', 'HPSv2', 'LPIPS', 'SSIM'])
    else:
        df = pd.read_csv('./results/results.csv')
    
    df.loc[len(df.index)] = [datetime.now().strftime("%d-%m-%Y--%H-%M-%S"), args.exp_name, inc_score, fid, clip_score, hps, lpips, ssim]
    df.to_csv('./results/results.csv', index=False)
