# Evaluation Suite
This repository contains code to easily evaluate results from diffusion models.
Originally this suite was developed to evaluate InsertDiffusion, so metrics might focus on aspects relevant to that project.
However, over metrics might be added in the future

**Work in Progress!**

Currently the implementation covers
- Inception Score (IS) for overall image quality
- Fréchet Inception Distance (FID) to compare visual similarity between to sets of images (typically ground truth and generated)
- CLIP score for prompt alignment
- Learned Perceptual Image Patch Similarity (LPIPS) to assess perceptual similarity of regions
- Structural Similarity Index (SSIM) to asses structrual similarity of regions
- Human Preference Score to assess perceived quality by humans

Planned to be added:
- Code to analyze results from a human evaluation study


## Usage
Please name your images by the prompt used following the scheme `<prompt>.png`.
Name reference images (i.e. the images from which the shape is taken as `<prompt>_<object type>.png`).
Move all images to the same folder e.g. `evaluation/result_images/exp_0`

### CLI
To evaluate your images you may easily use the command line interface like this
```bash
python evaluate.py --exp_name "your experiment name" --gen_file_path "<path to your generated images>"
```

Further, you may provide `--ref_file_path` for reference objects (e.g. if you inserted objects into another scene), `--gt_file_path` for ground truth images (e.g. to compare with real images), and `--masks_path` for mask images to tell the model where the inserted object is found in the generated images.

You may deactivate the calculation of specific metrics to save computation

| command    | metric                     |
|------------|----------------------------|
| --no_is    | Inception Score            |
| --no_fid   | Fréchet Inception Distance |
| --no_clip  | CLIP score                 |
| --no_hps   | Human Preference Score     |
| --no_lpips | LPIPS                      |
| --no_ssim  | SSIM                       |
