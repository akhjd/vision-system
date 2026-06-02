"""
Trains a small CNN to classify circle symbols into 6 classes.

Classes:
    0 - full white
    1 - full dark (teal)
    2 - half dark left
    3 - half dark right
    4 - half dark top
    5 - half dark bottom

Usage:
    python3 train.py
    python3 train.py --data data --epochs 30 --batch 64

Output:
    model_best.pth   best checkpoint (saved when val accuracy improves)
    model_final.pth  final checkpoint after all epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import argparse
import time


NUM_CLASSES = 6
IMAGE_SIZE  = 64
VAL_SPLIT   = 0.15


class CircleClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def is_valid_file(path):
    return "_preview" not in path


def load_datasets(data_dir):
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=train_transforms,
        is_valid_file=is_valid_file,
    )

    n_total = len(full_dataset)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    val_set.dataset = datasets.ImageFolder(
        root=data_dir,
        transform=val_transforms,
        is_valid_file=is_valid_file,
    )

    print(f"Dataset: {n_total} total  |  {n_train} train  |  {n_val} val")
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to index: {full_dataset.class_to_idx}\n")

    return train_set, val_set, full_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            for label, pred in zip(labels, preds):
                class_total[label.item()]   += 1
                class_correct[label.item()] += (pred == label).item()

    per_class = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(NUM_CLASSES)
    ]

    return total_loss / total, correct / total, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=str,   default="data")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--workers", type=int,   default=4)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    train_set, val_set, class_names = load_datasets(args.data)

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=False,
    )

    model = CircleClassifier(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4,
    )

    best_val_acc = 0.0
    class_labels = ["white", "dark", "half_L", "half_R", "half_T", "half_B"]

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, per_class = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
            f"{val_loss:>8.4f}  {val_acc:>6.1%}  {current_lr:>8.2e}  "
            f"({elapsed:.1f}s)"
        )

        if epoch % 5 == 0:
            print("  Per-class val acc:", end="")
            for name, acc in zip(class_labels, per_class):
                print(f"  {name}: {acc:.1%}", end="")
            print()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "class_names": class_names,
            }, "model_best.pth")
            print(f"  Best model saved (val acc: {val_acc:.1%})")

    torch.save({
        "epoch":       args.epochs,
        "model_state": model.state_dict(),
        "val_acc":     val_acc,
        "class_names": class_names,
    }, "model_final.pth")

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.1%}")
    print(f"Saved: model_best.pth, model_final.pth")


if __name__ == "__main__":
    main()