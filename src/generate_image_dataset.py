"""
generate_image_dataset.py
--------------------------
IMPORTANT / HONEST NOTE:
This sandbox has no internet access to download a real-world civic-complaint
image dataset (e.g. from Kaggle) and no access to ImageNet weight servers.
To still deliver a genuinely trainable, end-to-end image pipeline, this script
procedurally generates synthetic images with class-distinctive visual
patterns (colors, textures, shapes) for each of the 5 categories:

    Garbage            -> irregular brown/green/grey blobs on a ground texture
    Road Damage        -> grey asphalt background with dark jagged crack lines
    Water Leakage      -> blue/white radial puddle + streak patterns
    Drainage Blockage  -> dark murky irregular pool with grid (drain grate) lines
    Streetlight Fault  -> night-dark background with a pole and a lit/unlit bulb

This lets the CNN (MobileNetV2 architecture) actually learn and be evaluated
end-to-end (real train/val split, real accuracy). For production use, replace
the contents of data/images/<category>/ with real photographs of each issue
type (recommended: at least a few hundred real images per class) and re-run
train_image_model.py - no other code changes are required.

Run:
    python src/generate_image_dataset.py
Produces:
    data/images/<Category>/img_XXXX.png  (default 220 images per category)
"""

import os
import random
from PIL import Image, ImageDraw, ImageFilter

random.seed(7)

IMG_SIZE = 160
CATEGORIES = ["Garbage", "Road Damage", "Water Leakage", "Drainage Blockage", "Streetlight Fault"]


def _noisy_background(base_color, variance=15):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), base_color)
    draw = ImageDraw.Draw(img)
    for _ in range(400):
        x, y = random.randint(0, IMG_SIZE - 1), random.randint(0, IMG_SIZE - 1)
        r = max(0, min(255, base_color[0] + random.randint(-variance, variance)))
        g = max(0, min(255, base_color[1] + random.randint(-variance, variance)))
        b = max(0, min(255, base_color[2] + random.randint(-variance, variance)))
        draw.point((x, y), fill=(r, g, b))
    return img


def make_garbage_image():
    img = _noisy_background((90, 80, 60), variance=20)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(8, 14)):
        cx, cy = random.randint(10, IMG_SIZE - 10), random.randint(10, IMG_SIZE - 10)
        w, h = random.randint(10, 30), random.randint(10, 30)
        color = random.choice([(60, 45, 30), (100, 120, 40), (140, 140, 130), (70, 70, 30)])
        draw.ellipse([cx - w, cy - h, cx + w, cy + h], fill=color)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def make_road_damage_image():
    img = _noisy_background((110, 110, 112), variance=10)
    draw = ImageDraw.Draw(img)
    # jagged dark crack lines
    for _ in range(random.randint(3, 6)):
        x, y = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        for _ in range(random.randint(6, 12)):
            nx = x + random.randint(-15, 15)
            ny = y + random.randint(-15, 15)
            draw.line([x, y, nx, ny], fill=(30, 30, 30), width=random.randint(2, 4))
            x, y = nx, ny
    # a dark pothole blob
    cx, cy = random.randint(40, 120), random.randint(40, 120)
    r = random.randint(15, 30)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(25, 25, 25))
    return img.filter(ImageFilter.GaussianBlur(0.4))


def make_water_leakage_image():
    img = _noisy_background((150, 150, 140), variance=10)
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2 + random.randint(-20, 20), IMG_SIZE // 2 + random.randint(-20, 20)
    for r in range(60, 5, -6):
        shade = 150 + (60 - r)
        draw.ellipse([cx - r, cy - r * 0.6, cx + r, cy + r * 0.6],
                     outline=(80, 140, 200), width=2)
    draw.ellipse([cx - 25, cy - 15, cx + 25, cy + 15], fill=(120, 170, 220))
    for _ in range(random.randint(4, 8)):
        x = random.randint(0, IMG_SIZE)
        draw.line([x, 0, x + random.randint(-10, 10), IMG_SIZE], fill=(140, 190, 230), width=1)
    return img.filter(ImageFilter.GaussianBlur(0.5))


def make_drainage_blockage_image():
    img = _noisy_background((55, 60, 45), variance=12)
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    r = random.randint(45, 65)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(35, 40, 30))
    # grate lines
    for i in range(-r, r, 12):
        draw.line([cx - r, cy + i, cx + r, cy + i], fill=(20, 20, 15), width=2)
    for _ in range(random.randint(10, 18)):
        x = cx + random.randint(-r, r)
        y = cy + random.randint(-r, r)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(90, 100, 40))
    return img.filter(ImageFilter.GaussianBlur(0.5))


def make_streetlight_image():
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (10, 12, 25))
    draw = ImageDraw.Draw(img)
    for _ in range(150):
        x, y = random.randint(0, IMG_SIZE - 1), random.randint(0, IMG_SIZE - 1)
        draw.point((x, y), fill=(10 + random.randint(0, 15),) * 3)
    pole_x = IMG_SIZE // 2 + random.randint(-15, 15)
    draw.line([pole_x, IMG_SIZE, pole_x, 40], fill=(70, 70, 75), width=6)
    draw.line([pole_x, 40, pole_x + 25, 30], fill=(70, 70, 75), width=5)
    is_lit = random.random() < 0.5
    bulb_center = (pole_x + 25, 30)
    if is_lit:
        for r in range(28, 4, -4):
            shade = int(255 * (r / 28))
            draw.ellipse([bulb_center[0] - r, bulb_center[1] - r, bulb_center[0] + r, bulb_center[1] + r],
                         fill=(255, 240, min(200, shade)))
    else:
        draw.ellipse([bulb_center[0] - 6, bulb_center[1] - 6, bulb_center[0] + 6, bulb_center[1] + 6],
                     fill=(40, 40, 40), outline=(80, 80, 80))
    return img.filter(ImageFilter.GaussianBlur(0.3))


GENERATORS = {
    "Garbage": make_garbage_image,
    "Road Damage": make_road_damage_image,
    "Water Leakage": make_water_leakage_image,
    "Drainage Blockage": make_drainage_blockage_image,
    "Streetlight Fault": make_streetlight_image,
}


def main(images_per_category=220):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images"))
    for category in CATEGORIES:
        cat_dir = os.path.join(base_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        gen_fn = GENERATORS[category]
        for i in range(images_per_category):
            img = gen_fn()
            img.save(os.path.join(cat_dir, f"img_{i:04d}.png"))
        print(f"{category}: wrote {images_per_category} images to {cat_dir}")


if __name__ == "__main__":
    main(images_per_category=220)
