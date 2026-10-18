"""Versioned image views shared by feature extraction and inference."""
import hashlib
import json

import numpy as np
import torch
from PIL import Image, ImageOps

LEGACY = {'version': 1, 'size': 224, 'views': 1}
DETAIL = {'version': 2, 'size': 320, 'views': 2}


def validate_recipe(recipe):
    if recipe not in (LEGACY, DETAIL):
        raise ValueError('Unsupported image preprocessing recipe.')
    return recipe


def image_views(image, recipe):
    validate_recipe(recipe)
    image = ImageOps.exif_transpose(image).convert('RGB')
    if recipe == LEGACY:
        return [image]
    w, h = image.size
    dx, dy = int(w * .06), int(h * .06)
    detail = image.crop((dx, dy, w-dx, h-dy))
    return [ImageOps.pad(view, (320, 320), method=Image.Resampling.BICUBIC,
                         color=(124, 116, 104)) for view in (image, detail)]


def view_transform(recipe):
    from torchvision.models import EfficientNet_B0_Weights
    validate_recipe(recipe)
    return EfficientNet_B0_Weights.DEFAULT.transforms(
        crop_size=recipe['size'], resize_size=256 if recipe == LEGACY else 320)


def extract_features(model, images, recipe):
    from .food9 import visual_features
    transform = view_transform(recipe)
    views = [image_views(im, recipe) for im in images]
    with torch.inference_mode():
        batch = torch.stack([transform(view) for group in views for view in group])
        deep = model(batch).reshape(len(images), recipe['views'], -1).mean(1)
    # Keep the full original's colour distribution: padding is not food evidence.
    appearance = torch.from_numpy(np.stack([visual_features(im) for im in images]))
    return torch.cat((deep.cpu(), appearance), 1)


def feature_signature(rows, recipe):
    payload = {'images': [(r['sha256'], r['split']) for r in rows],
               'preprocessing': validate_recipe(recipe)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
