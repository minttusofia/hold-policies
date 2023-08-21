import functools
import hydra
import omegaconf
import r3m
import torch

import numpy as np
import torchvision.transforms as T

from PIL import Image

from hold_policies.dist_models import model as dist_model


class R3M(dist_model.EmbeddingFn):

    def __init__(self, model_name, distance_type='eucl', distance_params=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = r3m.load_r3m(model_name)
        model.eval()
        model.to(self.device)
        self.transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()])  # ToTensor() divides by 255
        embed_fn = functools.partial(self.embed, model)
        if distance_type != 'eucl':
            raise NotImplementerError(
                f'distance type {distance_type} not implemented for R3M')
        dist_fn = lambda s1, s2: np.linalg.norm((s1 - s2).cpu())
        super().__init__(embed_fn, dist_fn, history_length=1)

    def embed(self, model, img):
        if len(img.shape) > 3:
            img = np.squeeze(img, axis=3)
        img = (
            self.transforms(Image.fromarray((img * 255).astype(np.uint8)))
            .reshape(-1, 3, 224, 224))
        img = img.to(self.device)
        with torch.no_grad():
            emb = model(img * 255)
        return emb 
