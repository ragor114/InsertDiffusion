import torch
import torchvision.transforms as T
import clip
import hpsv2
import wget
import os
if os.getcwd().split('/')[-1] == 'metrics' or os.getcwd().split('\\')[-1] == 'metrics':
    os.chdir('..')
if os.getcwd().split('/')[-1] == '_notebooks' or os.getcwd().split('\\')[-1] == '_notebooks':
    os.chdir('..')


def load_model(model_path = './metrics/models/'):
    if not os.path.exists(model_path):
        os.makedirs('/'.join(model_path.split('/')[:-1]), exist_ok=True)
    if not os.path.exists(model_path + 'hps.pt'):
        wget.download('https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EWDmzdoqa1tEgFIGgR5E7gYBTaQktJcxoOYRoTHWzwzNcw?e=b7rgYW', out=model_path)
    model_path = os.path.join(model_path, 'hps.pt')

    model, preprocess = clip.load('ViT-L/14')
    model.load_state_dict(model_path)

    return model, preprocess


def _hps(imgs: list, prompts: list, model_path: str='./metrics/models/') -> float:
    global hps_model, hps_preprocess

    if hps_model is None:
        hps_model, hps_preprocess = load_model(model_path)
    
    scores = []
    for pil_img, prompt in zip(imgs, prompts):
        image = hps_preprocess(pil_img).unsqueeze(0)
        text = clip.tokenize([prompt])
        with torch.no_grad():
            image_features = hps_model.encode_image(image)
            text_features = hps_model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            hps = image_features @ text_features.T
            hps = hps.diagonal()
            scores.append(hps.squeeze().tolist())

    return sum(scores) / len(scores)

def hps(imgs: list, prompts: list) -> float:
    if type(prompts) == str:
        prompts = [prompts for _ in range(len(imgs))]

    scores = []
    
    for i, img in enumerate(imgs):
        score = hpsv2.score(img, prompts[i], hps_version="v2.1")[0]
        scores.append(score)
    return sum(scores)/len(scores)