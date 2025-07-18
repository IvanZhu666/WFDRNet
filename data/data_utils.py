import math
import cv2
import numpy as np
import torch
from imgaug import augmenters as iaa

def linear_interpolate(a, b, weight):
    return a + weight * (b - a)

def generate_perlin_noise(shape, res, fade_fn=lambda t: t**3 * (t * (t * 6 - 15) + 10)):
    step = (res[0] / shape[0], res[1] / shape[1])
    grid = np.mgrid[0:res[0]:step[0], 0:res[1]:step[1]].transpose(1, 2, 0) % 1

    angles = np.random.rand(res[0] + 1, res[1] + 1) * 2 * math.pi
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_gradients(r_slice, c_slice):
        grad = gradients[r_slice[0]:r_slice[1], c_slice[0]:c_slice[1]]
        grad = np.repeat(np.repeat(grad, shape[0] // res[0], axis=0), shape[1] // res[1], axis=1)
        return cv2.resize(grad, (shape[1], shape[0]))

    def dot_grid(grad, shift):
        shifted = grid + shift
        return (grad[:shape[0], :shape[1]] * shifted[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot_grid(tile_gradients((0, -1), (0, -1)), [0, 0])
    n10 = dot_grid(tile_gradients((1, None), (0, -1)), [-1, 0])
    n01 = dot_grid(tile_gradients((0, -1), (1, None)), [0, -1])
    n11 = dot_grid(tile_gradients((1, None), (1, None)), [-1, -1])

    fade_grid = fade_fn(grid[:shape[0], :shape[1]])
    lerp_x1 = linear_interpolate(n00, n10, fade_grid[..., 0])
    lerp_x2 = linear_interpolate(n01, n11, fade_grid[..., 0])

    perlin = linear_interpolate(lerp_x1, lerp_x2, fade_grid[..., 1])
    return perlin * math.sqrt(2)

rotate_augmenter = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

def perlin_noise(image, texture_image, aug_probability=1.0):
    image_np = np.asarray(image, dtype=np.float32)
    texture_np = np.asarray(texture_image, dtype=np.float32)

    height, width = image_np.shape[:2]
    scale_x = 2 ** torch.randint(0, 6, (1,)).item()
    scale_y = 2 ** torch.randint(0, 6, (1,)).item()

    noise = generate_perlin_noise((height, width), (scale_x, scale_y))
    noise = rotate_augmenter(image=noise)
    noise = np.expand_dims(noise, axis=-1)

    threshold_mask = (noise > 0.5).astype(np.float32)

    anomaly_region = (texture_np * threshold_mask) / 255.0
    clean_image = image_np / 255.0

    blend_factor = torch.rand(1).item() * 0.8
    blended_image = (
        clean_image * (1 - threshold_mask) +
        (1 - blend_factor) * anomaly_region +
        blend_factor * clean_image * threshold_mask
    ).astype(np.float32)

    if torch.rand(1).item() > aug_probability:
        return clean_image, np.zeros_like(threshold_mask).transpose(2, 0, 1)

    mask = threshold_mask.transpose(2, 0, 1)
    return blended_image, mask
