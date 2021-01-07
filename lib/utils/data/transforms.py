import numpy as np
from albumentations.core.composition import Compose


def rgb_transform(rgb_frame, albumentations_compose: Compose = None, **kwargs):
    # Augmentation
    if albumentations_compose:
        rgb_frame = albumentations_compose(image=rgb_frame)['image']

    # Transforms
    features_dict = {'rgb_frame': np.moveaxis(np.array(rgb_frame), -1, 0)}
    features_dict.update(kwargs)
    return features_dict
