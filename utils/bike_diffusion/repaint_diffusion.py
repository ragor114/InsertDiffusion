import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from scipy.ndimage import binary_dilation, binary_fill_holes

from .diffusion import compute_alpha
from .diffusion import get_config, get_beta_schedule, get_args
from .diffusion import load_model

CONFIG_PATH = './utils/bike_diffusion/configs/biked_256.yml'
CKPT_PATH = './utils/bike_diffusion/ckpts'

# implements RePaint algorithm
# in each denoising step of the diffusion process the masked region is noised according to the timestep and pasted into the noisy image, then DDIM denoising is applied by predicting the noise with the provided model
def inpainting_steps(original_images, seq, model, b, mask, device: torch.device, eta =0, gfn =False):
    with torch.no_grad():
        original_images = original_images.to(device)
        # print(type(original_images))
        # print(original_images.shape)
        b= b.to(device)
        try:
            n = original_images.size(0)
        except AttributeError:
            original_images = torch.stack(original_images)
            n = original_images.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        generated_parts_his = []
        known_parts_his = []
        mask = mask.to(device) 
        mask_inv = (1-mask).to(device)
        # perform T denoising steps 
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(original_images.device)
            next_t = (torch.ones(n) * j).to(original_images.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            if i == seq[-1]:
                if gfn:
                    xs = [torch.randn_like(original_images)]
                else:
                    xt = original_images * at.sqrt() + torch.randn_like(original_images) * (1.0 - at).sqrt()
                    xs = [xt]
            xt = xs[-1].to(device)
            et = model(xt, t).to(device)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next_denoised = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(original_images)
            generated_parts = [ single_xt_next_denoised * mask_inv for single_xt_next_denoised in xt_next_denoised]
            generated_parts_his.append(generated_parts)

            xt_next_forward =  original_images * at_next.sqrt() + torch.randn_like(original_images) * (1.0 - at_next).sqrt()
            known_parts = [ single_xt_next_forward * mask for single_xt_next_forward in xt_next_forward]
            known_parts_his.append(known_parts)

            xt_next = [ _generated_parts + _known_parts for (_generated_parts, _known_parts) in zip(generated_parts, known_parts)]
            xt_next = torch.stack(xt_next)
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds, generated_parts_his, known_parts_his

# mask dilation and filling of holes improved performance according to pretests
def mask_dilation(mask: np.array, dilation: int, fill_holes: bool=True) -> np.array:
    mask = mask.copy()
    if fill_holes:
        mask = 1-binary_fill_holes(1-mask)
    if dilation > 0:
        mask = 1-binary_dilation(1-mask, iterations=dilation)
    if fill_holes:
        mask = 1-binary_fill_holes(1-mask)
    return mask

# full inpainting procedure using the provided model
def inpaint(background: Image, mask: Image, device: torch.device, num_steps: int=50, ckpt_id: str='291000', dilation: int=0, fill_holes: bool=False, num_images: int=1):
    assert num_steps < 1000  # maximal amount of steps is determined by the number of steps the model is trained with
    # DDPM by Jiajae Fan accepts only single channel images
    background = background.convert('L')

    # normalization
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x / 255 * 2 - 1),
        transforms.Lambda(lambda x: x[None, :].float())
    ])
    x = transform(background)
    if num_images > 1:
        x = x.expand(num_images, -1, -1, -1)

    args = get_args(CONFIG_PATH, num_steps)
    config = get_config(args)

    # get noise schedule
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )

    # load model
    model = load_model(CKPT_PATH, CONFIG_PATH, config, ckpt_id)

    b = torch.from_numpy(betas).to('cpu').float()
    seq = range(0, 1000, num_steps)
    
    # normalize and dilate mask
    mask = (np.array(mask)/255).astype(float)
    mask = mask_dilation(mask, dilation, fill_holes)
    mask = torch.Tensor(mask)

    # generate masked images
    masked_x = []
    for img in x:
        masked_img = img.to(device) * mask.to(device)
        masked_x.append(masked_img)

    # generate images by RePaint algorithm
    input_images = x.clone()
    _, repainting_images, _, _ = inpainting_steps(input_images, seq, model, b, mask, device, eta =1, gfn = True)

    # return restults as PIL images
    if num_images == 1:
        return inpaint_tensor_to_image(repainting_images[0])
    
    return [inpaint_tensor_to_image(x) for x in repainting_images]

# denormalization for tensors and creation of image
def inpaint_tensor_to_image(img: torch.tensor) -> Image:
    img = img.detach().to('cpu').permute((1, 2, 0)).numpy()
    inpainted_im = np.squeeze(img.copy(), axis=2)
    inpainted_im += 1
    inpainted_im /=2
    inpainted_im = (inpainted_im*255)
    inpainted_im = Image.fromarray(inpainted_im).convert('RGB')
    return inpainted_im