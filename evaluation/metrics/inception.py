from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
import torch

# Inception scores, in general, require to be evaluated on many generated images (50000>) to give robust estimates.
def inception_score(imgs: list):
    inception = InceptionScore(normalize=False)
    
    transform = T.Compose([
        T.PILToTensor()
    ])

    imgs = [transform(x) for x in imgs]

    for img in imgs:
        inception.update(img.unsqueeze(0))
    result = inception.compute()
    return result[0].detach().item()

def frechet_inception_distance(gen_imgs: list, gt_imgs: list):
    transform = T.Compose([
        T.PILToTensor()
    ])

    gen_imgs = torch.stack([transform(x) for x in gen_imgs])
    gt_imgs = torch.stack([transform(x) for x in gt_imgs])

    assert gen_imgs.size(1) == 3
    assert gt_imgs.size(1) == 3

    fid = FrechetInceptionDistance(feature=64, normalize=False)
    fid.update(gen_imgs, real=False)
    fid.update(gt_imgs, real=True)

    return fid.compute().detach().item()
