import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

SEED = 42
PER_CLASS = 100
BATCH_SIZE = 64

np.random.seed(SEED)
torch.manual_seed(SEED)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cifar = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
labels = np.array(cifar.targets)

rng = np.random.RandomState(SEED)
chosen = []
for c in range(10):
    class_idx = np.where(labels == c)[0]
    chosen.append(rng.choice(class_idx, PER_CLASS, replace=False))
chosen = np.concatenate(chosen)
rng.shuffle(chosen)

subset = Subset(cifar, chosen.tolist())
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Identity()
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("device:", device)

all_feats = []
all_labels = []

with torch.no_grad():
    for i, (images, targets) in enumerate(loader):
        out = model(images.to(device))
        all_feats.append(out.cpu().numpy())
        all_labels.append(targets.numpy())
        print("batch", i + 1, "of", len(loader), end="\r")

X = np.concatenate(all_feats).astype(np.float32)
y = np.concatenate(all_labels)
print()
print("features:", X.shape)

X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)

X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED)

print("train", len(X_train), "| val", len(X_val), "| test", len(X_test))

np.savez_compressed(
    "features.npz",
    X_tr=X_train, y_tr=y_train,
    X_va=X_val, y_va=y_val,
    X_te=X_test, y_te=y_test
)

print("saved features.npz")
