#!/bin/bash
set -e

PROJECT=ui_segmentation_pipeline
IMAGE=ui-segmentation:latest
VENV=.venv

echo "📁 Creating project directory..."
mkdir -p $PROJECT
cd $PROJECT

# ===============================
# Python virtual environment
# ===============================
if [ ! -d "$VENV" ]; then
  echo "🐍 Creating Python virtual environment..."
  python3 -m venv $VENV
fi

source $VENV/bin/activate
pip install --upgrade pip setuptools wheel

# ===============================
# requirements.txt
# ===============================
cat << 'EOF' > requirements.txt
torch
torchvision
transformers
numpy
pillow
opencv-python
matplotlib
onnx
onnxruntime
EOF

pip install -r requirements.txt

# ===============================
# generate_synthetic_ui.py
# ===============================
cat << 'EOF' > generate_synthetic_ui.py
import os, random, numpy as np
from PIL import Image, ImageDraw, ImageFont

NUM_IMAGES = 500
W, H = 1280, 720
OUT = "synthetic_ui"
IMG = os.path.join(OUT, "images")
MSK = os.path.join(OUT, "masks")
os.makedirs(IMG, exist_ok=True)
os.makedirs(MSK, exist_ok=True)

LABELS = {"start":1, "settings":2, "quit":3}
TEXT = {"start":"START GAME", "settings":"SETTINGS", "quit":"QUIT"}

try:
    FONT = ImageFont.truetype("arial.ttf", 36)
except:
    FONT = ImageFont.load_default()

def background():
    img = np.zeros((H,W,3), dtype=np.uint8)
    if random.random() < 0.5:
        img[:] = np.random.randint(0,255,3)
    else:
        for y in range(H):
            c = int(255*y/H)
            img[y,:,:] = [c,c,c]
    return Image.fromarray(img)

def button(di, dm, label):
    bw, bh = random.randint(260,420), random.randint(70,110)
    x = random.randint(50, W-bw-50)
    y = random.randint(50, H-bh-50)
    di.rounded_rectangle([x,y,x+bw,y+bh], radius=18,
                         fill=tuple(np.random.randint(80,220,3)))
    dm.rectangle([x,y,x+bw,y+bh], fill=LABELS[label])
    t = TEXT[label]
    tw, th = di.textsize(t, font=FONT)
    di.text((x+(bw-tw)//2, y+(bh-th)//2),
            t, fill=(255,255,255), font=FONT)

for i in range(NUM_IMAGES):
    img = background()
    msk = Image.new("L", (W,H), 0)
    di, dm = ImageDraw.Draw(img), ImageDraw.Draw(msk)
    keys = list(LABELS.keys())
    random.shuffle(keys)
    for k in keys[:random.randint(1,3)]:
        button(di, dm, k)
    img.save(f"{IMG}/{i:04d}.png")
    msk.save(f"{MSK}/{i:04d}.png")
    if i % 50 == 0:
        print(f"{i}/{NUM_IMAGES}")

print("✅ Synthetic data generated")
EOF

# ===============================
# train_segformer.py
# ===============================
cat << 'EOF' > train_segformer.py
import os, cv2, torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH = 2
LR = 6e-5

class UIGameDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = sorted(os.listdir(root + "/images"))
        self.msks = sorted(os.listdir(root + "/masks"))

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.imread(f"{self.root}/images/{self.imgs[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(f"{self.root}/masks/{self.msks[i]}", cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img).permute(2,0,1).float()/255
        msk = torch.tensor(msk).long()
        return img, msk

ds = UIGameDataset("synthetic_ui")
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=4,
    ignore_mismatched_sizes=True
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR)

for e in range(EPOCHS):
    total = 0
    for img, msk in dl:
        img, msk = img.to(DEVICE), msk.to(DEVICE)
        out = model(pixel_values=img, labels=msk)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {e+1}/{EPOCHS} | Loss {total/len(dl):.4f}")

os.makedirs("checkpoints", exist_ok=True)
model.save_pretrained("checkpoints/segformer_ui")
print("✅ Model saved")
EOF

# ===============================
# export_onnx.py
# ===============================
cat << 'EOF' > export_onnx.py
import os, torch
from transformers import SegformerForSemanticSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("onnx", exist_ok=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    "checkpoints/segformer_ui"
).to(DEVICE).eval()

dummy = torch.randn(1, 3, 720, 1280).to(DEVICE)

torch.onnx.export(
    model,
    dummy,
    "onnx/segformer_ui.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}
)

print("✅ ONNX model exported")
EOF

# ===============================
# Dockerfile
# ===============================
cat << 'EOF' > Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["/bin/bash"]
EOF

# ===============================
# Build Docker image
# ===============================
docker build -t $IMAGE .

echo ""
echo "✅ Setup complete"
echo ""
echo "LOCAL (venv):"
echo "source .venv/bin/activate"
echo "python generate_synthetic_ui.py"
echo "python train_segformer.py"
echo "python export_onnx.py"
echo ""
echo "DOCKER:"
echo "docker run --gpus all -it -v \$(pwd):/app $IMAGE"

