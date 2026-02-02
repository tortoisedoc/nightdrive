#!/bin/bash
cat << 'EOF' > make_synthetic_ui_zip.sh
#!/bin/bash

set -e

PROJECT_DIR=synthetic_ui_generator
ZIP_NAME=synthetic_ui_generator.zip

mkdir -p $PROJECT_DIR

cat << 'PYCODE' > $PROJECT_DIR/generate_synthetic_ui.py
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NUM_IMAGES = 500
WIDTH, HEIGHT = 1280, 720

OUT_DIR = "synthetic_ui"
IMG_DIR = os.path.join(OUT_DIR, "images")
MASK_DIR = os.path.join(OUT_DIR, "masks")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

LABELS = {
    "start": 1,
    "settings": 2,
    "quit": 3
}

BUTTON_TEXT = {
    "start": "START GAME",
    "settings": "SETTINGS",
    "quit": "QUIT"
}

try:
    FONT = ImageFont.truetype("arial.ttf", 36)
except:
    FONT = ImageFont.load_default()

def random_background():
    bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    if random.random() < 0.5:
        bg[:] = np.random.randint(0, 255, size=3)
    else:
        for y in range(HEIGHT):
            c = int(255 * y / HEIGHT)
            bg[y, :, :] = [c, c, c]
    return Image.fromarray(bg)

def draw_button(draw_img, draw_mask, label):
    w = random.randint(260, 420)
    h = random.randint(70, 110)
    x = random.randint(50, WIDTH - w - 50)
    y = random.randint(50, HEIGHT - h - 50)

    color = tuple(np.random.randint(80, 220, size=3))
    radius = 18

    draw_img.rounded_rectangle(
        [x, y, x + w, y + h],
        radius=radius,
        fill=color
    )

    draw_mask.rectangle(
        [x, y, x + w, y + h],
        fill=LABELS[label]
    )

    text = BUTTON_TEXT[label]
    tw, th = draw_img.textsize(text, font=FONT)
    tx = x + (w - tw) // 2
    ty = y + (h - th) // 2
    draw_img.text((tx, ty), text, fill=(255,255,255), font=FONT)

for i in range(NUM_IMAGES):
    image = random_background()
    mask = Image.new("L", (WIDTH, HEIGHT), 0)

    draw_img = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)

    buttons = list(LABELS.keys())
    random.shuffle(buttons)

    for b in buttons[:random.randint(1,3)]:
        draw_button(draw_img, draw_mask, b)

    image.save(os.path.join(IMG_DIR, f"{i:04d}.png"))
    mask.save(os.path.join(MASK_DIR, f"{i:04d}.png"))

    if i % 50 == 0:
        print(f"Generated {i}/{NUM_IMAGES}")

print("✅ Dataset generation complete")
PYCODE

cat << 'README' > $PROJECT_DIR/README.md
# Synthetic UI Dataset Generator

## Install
pip install pillow numpy

## Run
python generate_synthetic_ui.py

## Output
synthetic_ui/
├── images/
└── masks/

Mask labels:
0 = background
1 = start game
2 = settings
3 = quit
README

zip -r $ZIP_NAME $PROJECT_DIR
echo "✅ Created $ZIP_NAME"
EOF

chmod +x make_synthetic_ui_zip.sh
./make_synthetic_ui_zip.sh
