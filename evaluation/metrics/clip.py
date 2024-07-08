import torch
from torchmetrics.multimodal.clip_score import CLIPScore
import torchvision.transforms as T

def clip_score(imgs: list, prompts: list) -> float:
    transform = T.Compose([
        T.PILToTensor()
    ])

    imgs = [transform(x) for x in imgs]

    if type(prompts) == str:
        prompts = [prompts for _ in range(imgs.size(0))]
    
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    clip.update(imgs, prompts)

    return clip.compute().detach().item()

