"""Task 4 — a better leaf classifier.

Starts from the Task 3 pipeline (the own Dataset of 3a, LeafNetBN of 3d, Adam 1e-3,
15 epochs, the geometric augmentation of 3d) and changes ONE thing per variant,
everything else identical, on the same training images with a fixed seed.

Three validation sets, built once and identical for every variant:
  val        the group's own validation photos             (29 images, 1 image = 3.4 %)
  camera     those same photos made camera-like once with a fixed seed: random crop,
             rotation and colour change, 8 versions each   (232 images, 1 image = 0.4 %)
  archive    last year's photos — other students, other phones (120 images, 0.8 %)

Usage (run from the folder holding this file; the data lives outside the repository):
    python task4.py --data ../_task4_data --out task4_results.json [--only name,...] [--quick]
"""
import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE, BATCH = 96, 32

# ----------------------------------------------------------------- data (Task 3a)

class LeafDataset(Dataset):
    """3a's Dataset: sorted subfolders are the classes, sorted files the samples.
    Allowed to span several folders, which is all 'add last year's photos' needs."""

    def __init__(self, folders, transform):
        roots = [Path(folders)] if isinstance(folders, str) else [Path(f) for f in folders]
        self.transform = transform
        self.classes = sorted(d.name for d in roots[0].iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [(str(p), self.class_to_idx[c])
                        for r in roots for c in self.classes
                        for p in sorted((r / c).glob("*")) if p.is_file()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), cls


def base_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def geo_transform():
    """The augmentation of 3d — geometry only."""
    return transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def geo_color_transform():
    """3d's augmentation plus colour jitter — the first Task 4 candidate."""
    return transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def camera_like(val_folder, versions=8, seed=1234):
    """A camera-like validation set made ONCE from the validation photos, with a fixed
    seed, so every variant is scored on exactly the same images. Stronger than the
    training augmentation: a phone camera shows a leaf off-centre, rotated, in shade.
    The seed is set here, and this runs before any training, so the sequence of random
    crops is the same for every variant and every run of this script."""
    torch.manual_seed(seed)
    tough = transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.35, 1.0), ratio=(0.7, 1.4)),
        transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.15)], p=0.9),
        transforms.RandomAffine(degrees=45, translate=(0.08, 0.08)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    images, labels = [], []
    classes = sorted(d.name for d in Path(val_folder).iterdir() if d.is_dir())
    for c in classes:
        for path in sorted(Path(val_folder).glob(f"{c}/*")):
            raw = Image.open(path).convert("RGB")
            for _ in range(versions):
                images.append(tough(raw))
                labels.append(classes.index(c))
    return TensorDataset(torch.stack(images), torch.tensor(labels))

# ----------------------------------------------------------------- networks (3b, 2b)

class LeafNet(nn.Module):
    """3b's network, with the two Task 4 options available: dropout in the head and
    global average pooling instead of flattening (gap=True)."""

    def __init__(self, dropout=0.0, gap=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = gap
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, 64) if gap else nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = self.pool(torch.relu(conv(x)))
        x = x.mean((2, 3)) if self.gap else x.flatten(1)
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


class LeafNetBN(LeafNet):
    """3d's LeafNet with batch normalization after every convolution."""

    def __init__(self, dropout=0.0, gap=False):
        super().__init__(dropout, gap)
        self.bn1, self.bn2 = nn.BatchNorm2d(16), nn.BatchNorm2d(32)
        self.bn3, self.bn4 = nn.BatchNorm2d(64), nn.BatchNorm2d(64)

    def forward(self, x):
        for conv, bn in ((self.conv1, self.bn1), (self.conv2, self.bn2),
                         (self.conv3, self.bn3), (self.conv4, self.bn4)):
            x = self.pool(torch.relu(bn(conv(x))))
        x = x.mean((2, 3)) if self.gap else x.flatten(1)
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def mobilenet(frozen=False, dropout=None):
    """2b's network: pretrained MobileNetV3-small, last layer replaced by 4 outputs.
    frozen=True trains only that layer; dropout raises the dropout already in its
    head (0.2) — replaced by type, because which index it sits on changes between
    torchvision versions."""
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    if dropout is not None:
        for i, layer in enumerate(model.classifier):
            if isinstance(layer, nn.Dropout):
                model.classifier[i] = nn.Dropout(dropout)
    if frozen:
        for p in model.features.parameters():
            p.requires_grad = False
    return model

# ----------------------------------------------------------------- the loop (3c)

def train(variant, seed, data, epochs, device, log=lambda *a: None):
    torch.manual_seed(seed)
    random.seed(seed)
    group_train = f"{data}/leaves/train"
    if variant["data"] == "group+archive":
        train_folder = [group_train, f"{data}/leaves_archive/train"]
    elif variant["data"] == "archive":
        train_folder = f"{data}/leaves_archive/train"
    else:
        train_folder = group_train
    train_loader = DataLoader(LeafDataset(train_folder, variant["transform"]()),
                              batch_size=BATCH, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))

    model = variant["model"]().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if variant.get("optimizer") == "adamw":
        optimizer = torch.optim.AdamW(params, lr=variant["lr"], weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(params, lr=variant["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=variant.get("label_smoothing", 0.0))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if variant.get("scheduler") == "cosine" else None)

    best, history, t0 = {"acc": -1.0, "state": None, "epoch": 0}, [], time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        if scheduler:
            scheduler.step()
        acc = evaluate(model, val_sets["val"], device)[0]
        history.append(round(acc, 4))
        if variant.get("keep") == "best" and acc > best["acc"]:
            best = {"acc": acc, "state": copy.deepcopy(model.state_dict()), "epoch": epoch + 1}
        log(f"    epoch {epoch + 1:2d}  loss {loss_sum / len(train_loader):.4f}  val {acc:.3f}")
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    out = {"variant": variant["name"], "seed": seed, "epochs": epochs,
           "seconds": round(time.time() - t0, 1),
           "trainable_parameters": sum(p.numel() for p in params),
           "train_images": len(train_loader.dataset),
           "label_smoothing": variant.get("label_smoothing", 0.0),
           "optimizer": variant.get("optimizer", "adam"), "lr": variant["lr"],
           "best_epoch": best["epoch"], "last_val_acc": history[-1], "history": history}
    for name in ("val", "camera", "archive"):
        acc, per_class, confusions = evaluate(model, val_sets[name], device)
        out[f"{name}_acc"] = round(acc, 4)
        out[f"{name}_per_class"] = {k: round(v, 3) for k, v in per_class.items()}
        if name in ("camera", "archive") and len(val_sets[name].dataset) >= 100:
            out[f"{name}_confusions"] = confusions
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, per_ok, per_n, conf = 0, {}, {}, {}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += (pred == labels).sum().item()
        for p, t in zip(pred.tolist(), labels.tolist()):
            per_n[t] = per_n.get(t, 0) + 1
            per_ok[t] = per_ok.get(t, 0) + int(p == t)
            if p != t:
                conf[f"{CLASSES[t]}->{CLASSES[p]}"] = conf.get(f"{CLASSES[t]}->{CLASSES[p]}", 0) + 1
    per_class = {CLASSES[c]: per_ok.get(c, 0) / per_n[c] for c in per_n}
    return correct / sum(per_n.values()), per_class, dict(sorted(conf.items()))

# ----------------------------------------------------------------- variants

def variants():
    """One change per row against the baseline; anything not given keeps the baseline's value."""

    def V(name, **kw):
        base = dict(name=name, data="group", transform=geo_transform,
                    model=lambda: LeafNetBN(), lr=1e-3, epochs=15)
        base.update(kw)
        return base

    return [
        V("baseline"),                                    # 3d exactly as it stands
        V("no_augmentation", transform=base_transform),   # what augmentation is worth
        V("colorjitter", transform=geo_color_transform),  # + colour augmentation
        V("leafnetbn_gap", model=lambda: LeafNetBN(gap=True)),      # GAP instead of flatten
        V("leafnetbn_dropout", model=lambda: LeafNetBN(dropout=0.3)),

        V("mobilenet_frozen", model=lambda: mobilenet(frozen=True)),   # only the head
        V("mobilenet_finetuned", model=lambda: mobilenet()),           # = Task 2's model
        V("mobilenet_dropout04", model=lambda: mobilenet(dropout=0.4)),
        V("mobilenet_cos30_color", model=lambda: mobilenet(), transform=geo_color_transform,
          scheduler="cosine", epochs=30),                              # scheduler + epochs
        V("mobilenet_adamw", model=lambda: mobilenet(), optimizer="adamw"),
        V("mobilenet_labelsmooth", model=lambda: mobilenet(), label_smoothing=0.1),
        V("mobilenet_best_epoch", model=lambda: mobilenet(), keep="best"),

        V("archive_only", model=lambda: mobilenet(), data="archive"),        # last year only
        V("group_plus_archive", model=lambda: mobilenet(), data="group+archive"),
        V("group_plus_archive_all", model=lambda: mobilenet(), data="group+archive",
          transform=geo_color_transform, optimizer="adamw", label_smoothing=0.1,
          scheduler="cosine", epochs=30),                        # all the winners together
    ]


# how many seeds each row gets; a row not listed runs once, with seed 0 (written out
# here so that "run once" is a decision on the record, not an oversight)
SEEDS = {"baseline": (0, 1, 2), "no_augmentation": (0, 1, 2), "colorjitter": (0, 1, 2),
         "leafnetbn_gap": (0, 1, 2), "leafnetbn_dropout": (0, 1, 2),
         "mobilenet_frozen": (0,), "mobilenet_finetuned": (0, 1, 2),
         "mobilenet_dropout04": (0,), "mobilenet_cos30_color": (0, 1, 2),
         "mobilenet_adamw": (0,), "mobilenet_labelsmooth": (0,),
         "mobilenet_best_epoch": (0,), "archive_only": (0,),
         "group_plus_archive": (0, 1, 2), "group_plus_archive_all": (0, 1, 2)}

# ----------------------------------------------------------------- main

val_sets, CLASSES = {}, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../_task4_data")
    ap.add_argument("--out", default="task4_results.json")
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true", help="2 epochs, 1 seed — smoke test")
    ap.add_argument("--force", action="store_true", help="rerun what --out already holds")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    global val_sets, CLASSES
    CLASSES = sorted(d.name for d in Path(f"{args.data}/leaves/val").iterdir() if d.is_dir())
    val_sets = {
        "val": DataLoader(LeafDataset(f"{args.data}/leaves/val", base_transform()), BATCH),
        "camera": DataLoader(camera_like(f"{args.data}/leaves/val"), BATCH),
        "archive": DataLoader(LeafDataset(f"{args.data}/leaves_archive/val", base_transform()), BATCH),
    }
    n = {k: len(v.dataset) for k, v in val_sets.items()}
    print(f"device {device}, threads {torch.get_num_threads()} | classes {CLASSES}")
    print(f"val {n['val']} | camera {n['camera']} | archive {n['archive']} images")
    print(f"noise: one val image = {100 / n['val']:.1f} %, "
          f"one camera image = {100 / n['camera']:.2f} %, one archive image = {100 / n['archive']:.1f} %\n")

    wanted = set(args.only.split(",")) if args.only else None
    results = [] if args.force or not Path(args.out).exists() \
        else json.load(open(args.out, encoding="utf-8"))
    done = set() if args.force else {(r["variant"], r["seed"], r["epochs"]) for r in results}
    if done:
        print(f"resuming, {len(done)} runs already in {args.out} (--force to rerun them)\n")
    header = (f"{'variant':<24}{'seed':>4}{'val':>7}{'camera':>8}{'archive':>9}"
              f"{'params':>9}{'train':>7}{'best':>6}{'s':>6}")
    print(header, "\n", "-" * len(header), sep="")
    for variant in variants():
        if wanted and variant["name"] not in wanted:
            continue
        for seed in ((0,) if args.quick else SEEDS.get(variant["name"], (0,))):
            epochs = 2 if args.quick else variant["epochs"]
            if (variant["name"], seed, epochs) in done:
                continue
            r = train(variant, seed, args.data, epochs, device,
                      log=print if wanted else (lambda *a: None))
            results.append(r)
            print(f"{r['variant']:<24}{r['seed']:>4}{r['val_acc']:>7.3f}{r['camera_acc']:>8.3f}"
                  f"{r['archive_acc']:>9.3f}{r['trainable_parameters']:>9d}{r['train_images']:>7d}"
                  f"{r['best_epoch'] or '-':>6}{r['seconds']:>6.0f}")
            json.dump(results, open(args.out, "w"), indent=1)   # keep partial results on disk
    print(f"\nwrote {args.out} ({len(results)} runs)")


if __name__ == "__main__":
    main()
