import numpy as np
from functools import partial
from copy import deepcopy
from torchvision.transforms import ToTensor
from numpy.random import random_sample

from breast_cancer_classifier.data_loading.loading import augment_and_normalize_image
from src.constants import VIEWS
from src.utils import imclahe, random_rotation
from src.breast_segmentation import ffdmForeground, get_contour


class MammoTransformation(object):
    """Data augmentation class for data loader
    """

    def __init__(self, parameters):
        random_number_generator = np.random.RandomState(parameters["seed"])
        # self.contours = parameters['contours']
        # self.flip_view = parameters['flip_view']
        # self.clahe = parameters['clahe']
        # self.std_noise = parameters['pixel_noise']
        self.aug = parameters["augmentation"]
        self.rotation = True if parameters["rotation"] != 0 else False
        if self.rotation:
            self.rotation_angle = parameters["rotation"]
        self.transformation_fnc = partial(
            augment_and_normalize_image,
            random_number_generator=random_number_generator,
            augmentation=parameters["augmentation"],
            max_crop_noise=parameters["max_crop_noise"],
            max_crop_size_noise=parameters["max_crop_size_noise"],
        )
        self.to_tensor = ToTensor() if parameters['to_tensor'] else None

    def __call__(self, im: np.ndarray, heatmap, info, view):
        cropped_image, cropped_heatmaps = self.transformation_fnc(
            image=im,
            auxiliary_image=heatmap,
            view=view,
            best_center=info['best_center'][view][0]
        )

        if heatmap is None:
            output = cropped_image[:, :, np.newaxis]
        else:
            output = np.concatenate([
                cropped_image[:, :, np.newaxis],
                cropped_heatmaps,
            ], axis=2)  #.transpose((2, 0, 1)) I comment this part because to_tensor transpose the dimensions

        if self.to_tensor is not None:
            output = self.to_tensor(output)

        # for view in VIEWS.LIST:
        #     im = sample[view]
        #     im = np.fliplr(im) if view[0] == self.flip_view else im
        #     if self.rotation and self.aug:
        #         im = random_rotation(im, self.rotation_angle)
        #     auxiliary_image = ffdmForeground(im)[:, :, np.newaxis] if self.contours else None
        #     output[view], mask = self.transformation_fnc(
        #         image=im,
        #         auxiliary_image=auxiliary_image,
        #         view=view,
        #         best_center=info['best_center'][view][0]
        #     )
        #     if self.clahe:
        #         clip_limit = 0.006 * random_sample() + 0.007 if self.aug else 0.01
        #         output[view] = imclahe(output[view], clip_limit=clip_limit)
        #     if self.contours:
        #         output['con_' + view] = get_contour(np.squeeze(mask, axis=2))
        #     if self.aug:
        #         noise = np.random.standard_normal(np.shape(output[view]))
        #         output[view] += self.std_noise * noise
        #
        #     if self.to_tensor is not None:
        #         output[view] = self.to_tensor(output[view][:, :, np.newaxis])
        #
        # output['output'] = np.asarray(sample['output']).astype(np.float32)

        return output
