import os
import re
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    import wandb
except Exception:
    wandb = None


ACTIONS = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
ACTION_TO_ID = {action: idx for idx, action in enumerate(ACTIONS)}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def measure_inference_time(model, loader, device, max_batches=None):
    model.eval()

    total_samples = 0
    total_time = 0.0

    for batch_idx, (clips, _) in enumerate(tqdm(loader, desc="Benchmark inference", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        clips = clips.to(device, non_blocking=(device.type == "cuda"))

        sync_device(device)
        start_time = time.perf_counter()
        _ = model(clips)
        sync_device(device)
        end_time = time.perf_counter()

        batch_size = clips.size(0)
        total_samples += batch_size
        total_time += end_time - start_time

    if total_samples == 0:
        return {
            "inference_time_sec": 0.0,
            "inference_ms_per_sample": 0.0,
            "inference_samples_per_sec": 0.0,
            "inference_samples": 0,
        }

    return {
        "inference_time_sec": float(total_time),
        "inference_ms_per_sample": float((total_time / total_samples) * 1000.0),
        "inference_samples_per_sec": float(total_samples / total_time),
        "inference_samples": int(total_samples),
    }


class ConvEncoder(nn.Module):
    """
    Per-frame CNN encoder.
    Input:  B*T, 3, 64, 64
    Output: B*T, encoder_dim
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


class CNNRNNActionModel(nn.Module):
    """
    CNN encoder + bidirectional GRU + temporal Conv1D classifier.
    """

    def __init__(
        self,
        num_classes: int = 6,
        encoder_dim: int = 256,
        rnn_hidden: int = 256,
        rnn_layers: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.encoder = ConvEncoder(out_dim=encoder_dim)

        self.rnn = nn.GRU(
            input_size=encoder_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        rnn_out_dim = 2 * rnn_hidden

        self.classifier = nn.Sequential(
            nn.Conv1d(rnn_out_dim, rnn_out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(rnn_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(rnn_out_dim, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, clip):

        b, t, c, h, w = clip.shape

        x = clip.reshape(b * t, c, h, w)
        x = self.encoder(x)
        x = x.reshape(b, t, -1)

        x, _ = self.rnn(x)

        # Conv1D expects B, channels, time.
        x = x.transpose(1, 2)
        logits = self.classifier(x)

        return logits


from pathlib import Path
import json
import torch
from torch.utils.data import Dataset


class PreprocessedKTHDatasetForPt(Dataset):
    def __init__(
        self,
        root,
        split,
        train_subjects=range(1, 17),
        val_subjects=range(17, 21),
        test_subjects=range(21, 26),
    ):
        """
        Dataset for preprocessed KTH .pt clips.

        """

        assert split in {"train", "val", "test"}

        self.root = Path(root)
        self.split = split

        metadata_path = self.root / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found at: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.actions = metadata.get(
            "actions",
            ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"],
        )

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
                f"No .pt samples found for split={split} under {self.root}. "
                f"Check metadata.json and subject_id values."
            )

        print(f"{split}: {len(self.samples)} .pt clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        sample_path = self.root / item["file"]

        if not sample_path.exists():
            raise FileNotFoundError(f"Missing sample file: {sample_path}")

        sample = torch.load(
            sample_path,
            map_location="cpu",
            weights_only=False,
        )

        clip = sample["frames"].float()
        label = torch.tensor(int(sample["label"]), dtype=torch.long)

        return clip, label

def train_one_epoch(model, loader, optimizer, scaler, device, epoch=None):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp = device.type == "cuda"

    total = 0
    correct = 0
    loss_sum = 0.0

    pbar = tqdm(
        loader,
        desc=f"Train {epoch}" if epoch is not None else "Train",
        leave=False,
    )

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


def save_history_csv(history, out_dir):
    history_df = pd.DataFrame(history)
    history_path = out_dir / "history.csv"
    history_df.to_csv(history_path, index=False)
    return history_path


def save_training_plots(history, out_dir):
    history_df = pd.DataFrame(history)

    acc_path = out_dir / "accuracy.png"
    loss_path = out_dir / "loss.png"

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_acc"], label="train acc")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("KTH Action Recognition Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("KTH Action Recognition Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close()

    return acc_path, loss_path


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


def save_confusion_matrix(y_true, y_pred, actions, out_dir, normalize=None):


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
        "Confusion Matrix Normalized"
        if normalize == "true"
        else "Confusion Matrix Counts"
    )

    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix CSV:  {cm_csv_path}")
    print(f"Saved confusion matrix plot: {cm_png_path}")

    return {
        "csv": str(cm_csv_path),
        "png": str(cm_png_path),
    }


def save_test_artifacts(y_true, y_pred, actions, out_dir):
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

    report_txt_path = out_dir / "classification_report.txt"
    report_csv_path = out_dir / "classification_report.csv"

    with open(report_txt_path, "w") as f:
        f.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(report_csv_path)

    print(report_text)

    return {
        "classification_report_txt": str(report_txt_path),
        "classification_report_csv": str(report_csv_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed-root",
        type=str,
        default="/home/nfs/data/hdd_datasets/kth_actions/processed",
        help="Path to kth_actions/processed",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--out-root", type=str, default="runs")
    parser.add_argument("--use-tensorboard", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="kth-action-recognition")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--inference-benchmark-batches", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    seed_everything(args.seed)

    device = get_device()
    print("Device:", device)

    if args.run_name is None:
        run_name = datetime.now().strftime("kth_run_%Y%m%d_%H%M%S")
    else:
        run_name = args.run_name

    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {out_dir}")

    pin_memory = device.type == "cuda"

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

    print(f"Train clips: {len(train_ds)}")
    print(f"Val clips:   {len(val_ds)}")
    print(f"Test clips:  {len(test_ds)}")

    model = CNNRNNActionModel(
        num_classes=len(ACTIONS),
        dropout=0.4,
    ).to(device) 
    
    num_learnable_params = count_learnable_parameters(model)
    print(f"Learnable parameters: {num_learnable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
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
        "model": "CNNRNNActionModel",
        "dataset": "KTH Actions processed PNG frames",
        "processed_root": args.processed_root,
        "actions": ACTIONS,
        "seq_len": args.seq_len,
        "window_stride": args.window_stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "use_amp": use_amp,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "learnable_parameters": int(num_learnable_params),
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Saved config to: {out_dir / 'config.json'}")

    writer = None
    if args.use_tensorboard:
        if SummaryWriter is None:
            raise ImportError("TensorBoard logging requested, but torch.utils.tensorboard is unavailable. Install tensorboard.")
        writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
        writer.add_text("config", json.dumps(config, indent=2))

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("W&B logging requested, but wandb is not installed. Run: pip install wandb")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config,
            dir=str(out_dir),
        )

    if writer is not None:
        writer.add_scalar("model/learnable_parameters", num_learnable_params, 0)
    if wandb_run is not None:
        wandb.log({"model/learnable_parameters": num_learnable_params}, step=0)

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
        "epoch_time_sec": [],
    }

    training_start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()

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
        epoch_time_sec = time.perf_counter() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time_sec)

        save_history_csv(history, out_dir)
        save_training_plots(history, out_dir)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
            f"lr {current_lr:.8f} | "
            f"epoch time {epoch_time_sec:.2f}s"
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("accuracy/train", train_metrics["acc"], epoch)
            writer.add_scalar("accuracy/val", val_metrics["acc"], epoch)
            writer.add_scalar("time/epoch_sec", epoch_time_sec, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "loss/train": train_metrics["loss"],
                "loss/val": val_metrics["loss"],
                "accuracy/train": train_metrics["acc"],
                "accuracy/val": val_metrics["acc"],
                "time/epoch_sec": epoch_time_sec,
                "lr": current_lr,
            }, step=epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "actions": ACTIONS,
            "epoch": epoch,
            "val_acc": val_metrics["acc"],
            "best_val_acc": best_val_acc,
            "config": config,
            "history": history,
        }

        torch.save(checkpoint, last_path)

        PATIENCE = 4
        epochs_without_improvement = 0

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            epochs_without_improvement = 0

            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, best_path)

            print(f"Saved new best model: {best_path} | val acc: {best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{PATIENCE} epochs")

            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

    total_training_time_sec = time.perf_counter() - training_start_time
    print(f"Total training time: {total_training_time_sec:.2f}s")

    print("\nLoading best checkpoint...")

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved. Check validation loop.")

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
        max_batches=args.inference_benchmark_batches,
    )

    y_true, y_pred = predict_all(model, test_loader, device)

    cm_counts_paths = save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        actions=ACTIONS,
        out_dir=out_dir,
        normalize=None,
    )

    cm_norm_paths = save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        actions=ACTIONS,
        out_dir=out_dir,
        normalize="true",
    )

    test_artifacts = save_test_artifacts(
        y_true=y_true,
        y_pred=y_pred,
        actions=ACTIONS,
        out_dir=out_dir,
    )

    results = {
        "best_val_acc": float(checkpoint["best_val_acc"]),
        "best_epoch": int(checkpoint["epoch"]),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "learnable_parameters": int(num_learnable_params),
        "total_training_time_sec": float(total_training_time_sec),
        "inference_time_sec": inference_metrics["inference_time_sec"],
        "inference_ms_per_sample": inference_metrics["inference_ms_per_sample"],
        "inference_samples_per_sec": inference_metrics["inference_samples_per_sec"],
        "inference_samples": inference_metrics["inference_samples"],
        "best_model_path": str(best_path),
        "last_model_path": str(last_path),
        "confusion_matrix_counts_csv": cm_counts_paths["csv"],
        "confusion_matrix_counts_png": cm_counts_paths["png"],
        "confusion_matrix_normalized_csv": cm_norm_paths["csv"],
        "confusion_matrix_normalized_png": cm_norm_paths["png"],
        "classification_report_txt": test_artifacts["classification_report_txt"],
        "classification_report_csv": test_artifacts["classification_report_csv"],
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    comparison_row = {
        "model": "CNNRNNActionModel",
        "best_val_acc": results["best_val_acc"],
        "test_acc": results["test_acc"],
        "test_loss": results["test_loss"],
        "learnable_parameters": results["learnable_parameters"],
        "total_training_time_sec": results["total_training_time_sec"],
        "inference_time_sec": results["inference_time_sec"],
        "inference_ms_per_sample": results["inference_ms_per_sample"],
        "inference_samples_per_sec": results["inference_samples_per_sec"],
    }
    comparison_df = pd.DataFrame([comparison_row])
    comparison_df.to_csv(out_dir / "single_model_comparison.csv", index=False)

    if writer is not None:
        writer.add_scalar("test/loss", results["test_loss"], results["best_epoch"])
        writer.add_scalar("test/accuracy", results["test_acc"], results["best_epoch"])
        writer.add_scalar("time/total_training_sec", results["total_training_time_sec"], 0)
        writer.add_scalar("time/inference_ms_per_sample", results["inference_ms_per_sample"], 0)
        writer.add_scalar("time/inference_samples_per_sec", results["inference_samples_per_sec"], 0)
        writer.flush()
        writer.close()

    if wandb_run is not None:
        wandb.log({
            "test/loss": results["test_loss"],
            "test/accuracy": results["test_acc"],
            "time/total_training_sec": results["total_training_time_sec"],
            "time/inference_ms_per_sample": results["inference_ms_per_sample"],
            "time/inference_samples_per_sec": results["inference_samples_per_sec"],
            "model/learnable_parameters": results["learnable_parameters"],
        })
        wandb_run.finish()

    print("=" * 60)
    print(f"Best val acc: {results['best_val_acc']:.4f}")
    print(f"Best epoch:   {results['best_epoch']}")
    print(f"Test loss:    {results['test_loss']:.4f}")
    print(f"Test acc:     {results['test_acc']:.4f}")
    print(f"Parameters:   {results['learnable_parameters']:,}")
    print(f"Train time:   {results['total_training_time_sec']:.2f}s")
    print(f"Infer time:   {results['inference_time_sec']:.2f}s")
    print(f"Infer/sample: {results['inference_ms_per_sample']:.4f} ms")
    print(f"Saved to:     {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
