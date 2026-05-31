import numpy as np
import cv2
from PIL import Image


def image_translation(img, params):
    dx, dy = params
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)


def image_scale(img, scale_factor):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale_factor)
    return cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)


def image_rotation(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)


def image_shear(img, shear_factor):
    rows, cols = img.shape[:2]
    M = np.float32([[1, shear_factor/10, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)


def image_contrast(img, factor):
    mean = np.mean(img)
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.float32)


def image_brightness(img, delta):
    return np.clip(img + delta, 0, 255).astype(np.float32)


def image_blur(img, ksize):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def image_noise(img, std=10):
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.float32)


def image_flip_horizontal(img):
    return cv2.flip(img, 1)


def image_flip_vertical(img):
    return cv2.flip(img, 0)
