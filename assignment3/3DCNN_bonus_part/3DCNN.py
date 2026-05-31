import os
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


ACTIONS = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PreprocessedKTHDatasetForPt(Dataset):
    def __init__(
        self,
        root,
        split,
        train_subjects=range(1, 17),
        val_subjects=range(17, 21),
        test_subjects=range(21, 26),
    ):
        assert split in {"train", "val", "test"}

        self.root = Path(root)
        self.split = split

        metadata_path = self.root / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found at: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.actions = metadata.get("actions", ACTIONS)
        all_samples = metadata["samples"]

        if split == "train":
            subject_ids = set(train_subjects)
        elif split == "val":
            subject_ids = set(val_subjects)
        else:
            subject_ids = set(test_subjects)

        self.samples = [
            sample for sample in all_samples
            if int(sample["subject_id"]) in subject_ids
        ]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .pt samples found for split={split} under {self.root}."
            )

        print(f"{split}: {len(self.samples)} .pt clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        sample_path = self.root / item["file"]

        sample = torch.load(
            sample_path,
            map_location="cpu",
            weights_only=False,
        )

        clip = sample["frames"].float()
        label = torch.tensor(int(sample["label"]), dtype=torch.long)

        return clip, label


class Basic3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.dropout = nn.Dropout3d(dropout)

        if in_channels != out_channels or stride != (1, 1, 1):
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class KTH3DCNN(nn.Module):


    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(
                3,
                32,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = Basic3DBlock(
            32,
            64,
            stride=(1, 2, 2),
            dropout=dropout,
        )

        self.layer2 = Basic3DBlock(
            64,
            128,
            stride=(2, 2, 2),
            dropout=dropout,
        )

        self.layer3 = Basic3DBlock(
            128,
            256,
            stride=(2, 2, 2),
            dropout=dropout,
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, clip):
        # Input: B, T, C, H, W
        # Conv3d expects: B, C, T, H, W
        x = clip.permute(0, 2, 1, 3, 4).contiguous()

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)

        logits = self.classifier(x)

        return logits


def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch=None):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    use_amp = device.type == "cuda"

    total = 0
    correct = 0
    loss_sum = 0.0

    pbar = tqdm(loader, desc=f"Train {epoch}" if epoch else "Train", leave=False)

    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(clips)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(clips)
            loss = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.numel()
        loss_sum += loss.item() * labels.numel()

        pbar.set_postfix(
            loss=f"{loss_sum / total:.4f}",
            acc=f"{correct / total:.4f}",
        )

    return {
        "loss": loss_sum / total,
        "acc": correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, device, desc="Eval"):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0

    pbar = tqdm(loader, desc=desc, leave=False)

    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))

        logits = model(clips)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.numel()
        loss_sum += loss.item() * labels.numel()

        pbar.set_postfix(
            loss=f"{loss_sum / total:.4f}",
            acc=f"{correct / total:.4f}",
        )

    return {
        "loss": loss_sum / total,
        "acc": correct / total,
    }


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for clips, labels in tqdm(loader, desc="Predict", leave=False):
        clips = clips.to(device, non_blocking=(device.type == "cuda"))

        logits = model(clips)
        preds = logits.argmax(dim=1).detach().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def measure_inference_time(model, loader, device, warmup_batches=5):
    model.eval()

    for batch_idx, (clips, _) in enumerate(loader):
        clips = clips.to(device, non_blocking=(device.type == "cuda"))
        _ = model(clips)

        if batch_idx + 1 >= warmup_batches:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    total_samples = 0

    for clips, _ in loader:
        clips = clips.to(device, non_blocking=(device.type == "cuda"))
        _ = model(clips)
        total_samples += clips.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start_time

    return {
        "inference_time_sec": total_time,
        "inference_ms_per_sample": 1000.0 * total_time / total_samples,
        "inference_samples_per_sec": total_samples / total_time,
    }


def save_training_plots(history, out_dir):
    history_df = pd.DataFrame(history)

    acc_path = out_dir / "accuracy.png"
    loss_path = out_dir / "loss.png"

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_acc"], label="train acc")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("3D-CNN KTH Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("3D-CNN KTH Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(model, loader, device, actions, out_dir, normalize=None):
    y_true, y_pred = predict_all(model, loader, device)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(actions))),
        normalize=normalize,
    )

    suffix = "normalized" if normalize == "true" else "counts"

    cm_csv_path = out_dir / f"confusion_matrix_{suffix}.csv"
    cm_png_path = out_dir / f"confusion_matrix_{suffix}.png"

    pd.DataFrame(
        cm,
        index=actions,
        columns=actions,
    ).to_csv(cm_csv_path)

    fig, ax = plt.subplots(figsize=(8, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=actions,
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        values_format=".2f" if normalize == "true" else "d",
        colorbar=True,
    )

    ax.set_title(
        "3D-CNN Confusion Matrix Normalized"
        if normalize == "true"
        else "3D-CNN Confusion Matrix Counts"
    )

    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "csv": str(cm_csv_path),
        "png": str(cm_png_path),
    }


def save_classification_report(model, loader, device, actions, out_dir):
    y_true, y_pred = predict_all(model, loader, device)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=actions,
        digits=4,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=actions,
        digits=4,
        output_dict=True,
    )

    txt_path = out_dir / "classification_report.txt"
    csv_path = out_dir / "classification_report.csv"

    with open(txt_path, "w") as f:
        f.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(csv_path)

    print(report_text)

    return {
        "classification_report_txt": str(txt_path),
        "classification_report_csv": str(csv_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--processed-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    seed_everything(args.seed)

    device = get_device()
    print("Device:", device)

    run_name = args.run_name
    if run_name is None:
        run_name = datetime.now().strftime("kth_3dcnn_%Y%m%d_%H%M%S")

    out_dir = Path("runs_3dcnn") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PreprocessedKTHDatasetForPt(
        root=args.processed_root,
        split="train",
    )

    val_ds = PreprocessedKTHDatasetForPt(
        root=args.processed_root,
        split="val",
    )

    test_ds = PreprocessedKTHDatasetForPt(
        root=args.processed_root,
        split="test",
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = KTH3DCNN(
        num_classes=len(ACTIONS),
        dropout=args.dropout,
    ).to(device)

    num_params = count_learnable_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    config = {
        "model": "KTH3DCNN",
        "dataset": "KTH preprocessed .pt clips",
        "processed_root": args.processed_root,
        "actions": ACTIONS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "device": str(device),
        "use_amp": use_amp,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "learnable_parameters": num_params,
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Saving outputs to: {out_dir}")
    print(f"Learnable parameters: {num_params:,}")

    best_val_acc = 0.0
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    train_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            desc=f"Val {epoch}",
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(current_lr)

        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        save_training_plots(history, out_dir)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
            f"lr {current_lr:.8f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "actions": ACTIONS,
            "epoch": epoch,
            "val_acc": val_metrics["acc"],
            "best_val_acc": best_val_acc,
            "num_params": num_params,
            "config": config,
            "history": history,
        }

        torch.save(checkpoint, last_path)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, best_path)

            print(f"Saved new best 3D-CNN: {best_path} | val acc: {best_val_acc:.4f}")

    total_training_time = time.time() - train_start_time

    print("\nLoading best checkpoint...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        desc="Test",
    )

    inference_metrics = measure_inference_time(
        model=model,
        loader=test_loader,
        device=device,
    )

    cm_counts_paths = save_confusion_matrix(
        model=model,
        loader=test_loader,
        device=device,
        actions=ACTIONS,
        out_dir=out_dir,
        normalize=None,
    )

    cm_norm_paths = save_confusion_matrix(
        model=model,
        loader=test_loader,
        device=device,
        actions=ACTIONS,
        out_dir=out_dir,
        normalize="true",
    )

    report_paths = save_classification_report(
        model=model,
        loader=test_loader,
        device=device,
        actions=ACTIONS,
        out_dir=out_dir,
    )

    results = {
        "model": "KTH3DCNN",
        "best_val_acc": float(checkpoint["best_val_acc"]),
        "best_epoch": int(checkpoint["epoch"]),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "learnable_parameters": int(num_params),
        "total_training_time_sec": float(total_training_time),
        "inference_time_sec": float(inference_metrics["inference_time_sec"]),
        "inference_ms_per_sample": float(inference_metrics["inference_ms_per_sample"]),
        "inference_samples_per_sec": float(inference_metrics["inference_samples_per_sec"]),
        "best_model_path": str(best_path),
        "last_model_path": str(last_path),
        "confusion_matrix_counts_csv": cm_counts_paths["csv"],
        "confusion_matrix_counts_png": cm_counts_paths["png"],
        "confusion_matrix_normalized_csv": cm_norm_paths["csv"],
        "confusion_matrix_normalized_png": cm_norm_paths["png"],
        "classification_report_txt": report_paths["classification_report_txt"],
        "classification_report_csv": report_paths["classification_report_csv"],
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    comparison_df = pd.DataFrame([results])
    comparison_df.to_csv(out_dir / "model_summary_3dcnn.csv", index=False)

    print("=" * 60)
    print("3D-CNN Final Results")
    print(f"Best val acc: {results['best_val_acc']:.4f}")
    print(f"Best epoch:   {results['best_epoch']}")
    print(f"Test loss:    {results['test_loss']:.4f}")
    print(f"Test acc:     {results['test_acc']:.4f}")
    print(f"Parameters:   {results['learnable_parameters']:,}")
    print(f"Train time:   {results['total_training_time_sec']:.2f}s")
    print(f"Infer/sample: {results['inference_ms_per_sample']:.4f} ms")
    print(f"Saved to:     {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()