import math
import random

import numpy as np
from PIL import Image, ImageDraw

from config import (
    IMAGE_SIZE,
    MIN_NUM_RECTS,
    MAX_NUM_RECTS,
    ROT_DEG_AUG,
    MIN_RECT_SIZE,
    MAX_RECT_SIZE,
)


def random_color():
    return tuple(np.random.randint(0, 256, size=3).tolist())


def rotated_bbox_size(w, h, angle_deg):
    angle = math.radians(angle_deg)

    rot_w = abs(w * math.cos(angle)) + abs(h * math.sin(angle))
    rot_h = abs(w * math.sin(angle)) + abs(h * math.cos(angle))

    return rot_w, rot_h


def random_valid_center(IMAGE_SIZE, w, h, angle_deg):
    rot_w, rot_h = rotated_bbox_size(w, h, angle_deg)

    margin_x = math.ceil(rot_w / 2)
    margin_y = math.ceil(rot_h / 2)

    if margin_x >= IMAGE_SIZE / 2 or margin_y >= IMAGE_SIZE / 2:
        raise ValueError("Rectangle is too large to fit inside the image.")

    cx = random.randint(margin_x, IMAGE_SIZE - margin_x)
    cy = random.randint(margin_y, IMAGE_SIZE - margin_y)

    return cx, cy


def paste_rotated_rect(base_img, cx, cy, w, h, color, angle):
    # Make a transparent patch containing the rectangle
    patch_size = math.ceil(math.sqrt(w**2 + h**2))
    patch = Image.new("RGBA", (patch_size, patch_size), (0, 0, 0, 0))

    draw = ImageDraw.Draw(patch)

    x0 = (patch_size - w) // 2
    y0 = (patch_size - h) // 2
    x1 = x0 + w
    y1 = y0 + h

    draw.rectangle([x0, y0, x1, y1], fill=color + (255,))

    # Rotate patch
    rotated = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    # Paste centered at (cx, cy)
    px = int(cx - rotated.width / 2)
    py = int(cy - rotated.height / 2)

    base_img.paste(rotated, (px, py), rotated)


def bbox_from_center(cx, cy, w, h, angle_deg):
    bbox_w, bbox_h = rotated_bbox_size(w, h, angle_deg)

    x_min = int(round(cx - bbox_w / 2))
    y_min = int(round(cy - bbox_h / 2))
    x_max = int(round(cx + bbox_w / 2))
    y_max = int(round(cy + bbox_h / 2))

    return [x_min, y_min, x_max, y_max]


def generate_image_and_labels():
    bg_pixels = np.random.randint(
        0,
        256,
        size=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )

    img = Image.fromarray(bg_pixels, mode="RGB")
    labels = []

    num_rects = random.randint(MIN_NUM_RECTS, MAX_NUM_RECTS)

    for _ in range(num_rects):
        w = random.randint(MIN_RECT_SIZE, MAX_RECT_SIZE)
        h = random.randint(MIN_RECT_SIZE, MAX_RECT_SIZE)

        color = random_color()
        angle = random.uniform(-ROT_DEG_AUG, ROT_DEG_AUG)

        cx, cy = random_valid_center(IMAGE_SIZE, w, h, angle)

        bbox = bbox_from_center(cx, cy, w, h, angle)

        paste_rotated_rect(img, cx, cy, w, h, color, angle)

        labels.append(
            {
                "bbox": bbox,  # [x_min, y_min, x_max, y_max]
                "angle": angle,
                "color": color,
            }
        )

    return img, labels
