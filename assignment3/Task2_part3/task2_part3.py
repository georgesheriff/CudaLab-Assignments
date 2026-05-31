import json
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display

DATA_ROOT = "KTH_preprocessed_augmented" 


def _train_epoch_task3(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start = time.time()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return {"loss": total_loss / total, "acc": correct / total, "time": time.time() - start}


@torch.no_grad()
def _eval_epoch_task3(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    start = time.time()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return {"loss": total_loss / total, "acc": correct / total, "time": time.time() - start}


def _split_name(subject_id):
    if subject_id <= 16:
        return "train"
    if subject_id <= 20:
        return "val"
    return "test"


SPLITS = {"train": [], "val": [], "test": []}
meta_path = Path(DATA_ROOT) / "metadata.json"
if meta_path.exists():
    for row in json.load(open(meta_path))["samples"]:
        SPLITS[_split_name(row["subject_id"])].append(Path(DATA_ROOT) / row["file"])
else:
    for path in sorted(Path(DATA_ROOT).glob("sample_*.pt")):
        sid = torch.load(path, map_location="cpu", weights_only=False)["subject_id"]
        SPLITS[_split_name(sid)].append(path)

print(
    f"samples: train {len(SPLITS['train'])} | "
    f"val {len(SPLITS['val'])} | test {len(SPLITS['test'])}"
)


def make_loader(split, batch_size=32, shuffle=False, num_workers=0):
    paths = SPLITS[split]

    class _DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            sample = torch.load(paths[i], map_location="cpu", weights_only=False)
            x = sample["frames"]  # (T, 3, 64, 64), already preprocessed
            x = x.mean(dim=1, keepdim=True)  # (T, 1, 64, 64) for 1-ch encoders above
            y = sample["label"]
            return x, torch.tensor(y, dtype=torch.long)

    return DataLoader(
        _DS(),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def measure_inference_time(model, loader, device, max_batches=50):
    model.eval()
    batches = 0
    samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for x, _ in loader:
        x = x.to(device)
        _ = model(x)
        batches += 1
        samples += x.size(0)
        if batches >= max_batches:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return {
        "inference_sec": elapsed,
        "samples": samples,
        "ms_per_sample": 1000.0 * elapsed / max(samples, 1),
    }


def run_experiment_task3(
    model_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=10,
    lr=3e-4,
    log_dir="runs/kth_actions",
):
    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.2)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_scalar("model/params", num_params, 0)

    best_val_acc = 0.0
    total_train_sec = 0.0

    for epoch in range(epochs):
        train_stats = _train_epoch_task3(model, train_loader, optimizer, criterion, device)
        val_stats = _eval_epoch_task3(model, val_loader, criterion, device)
        scheduler.step()

        total_train_sec += train_stats["time"]

        writer.add_scalar("loss/train", train_stats["loss"], epoch)
        writer.add_scalar("loss/val", val_stats["loss"], epoch)
        writer.add_scalar("accuracy/train", train_stats["acc"], epoch)
        writer.add_scalar("accuracy/val", val_stats["acc"], epoch)
        writer.add_scalar("time/train_epoch_sec", train_stats["time"], epoch)
        writer.add_scalar("time/val_epoch_sec", val_stats["time"], epoch)

        print(
            f"{model_name} | epoch {epoch + 1}/{epochs} | "
            f"train acc {train_stats['acc']:.4f} | val acc {val_stats['acc']:.4f}"
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pth")

    ckpt_path = f"checkpoints/{model_name}_best.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_stats = _eval_epoch_task3(model, test_loader, criterion, device)
    infer_stats = measure_inference_time(model, test_loader, device)

    writer.add_scalar("accuracy/test", test_stats["acc"], 0)
    writer.add_scalar("loss/test", test_stats["loss"], 0)
    writer.add_scalar("time/test_sec", test_stats["time"], 0)
    writer.add_scalar("time/total_train_sec", total_train_sec, 0)
    writer.add_scalar("time/inference_ms_per_sample", infer_stats["ms_per_sample"], 0)
    writer.close()

    return {
        "model": model_name,
        "params": num_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_stats["acc"],
        "test_loss": test_stats["loss"],
        "train_time_sec": total_train_sec,
        "test_eval_sec": test_stats["time"],
        "inference_ms_per_sample": infer_stats["ms_per_sample"],
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10

train_loader = make_loader("train", BATCH_SIZE, shuffle=True)
val_loader = make_loader("val", BATCH_SIZE)
test_loader = make_loader("test", BATCH_SIZE)

print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

models_to_compare = {
    "pytorch_lstmcell": ActionModelLSTMCell(),
    "pytorch_grucell": ActionModelGRUCell(),
    "own_lstm": ActionModelMyLSTM(),
    "own_convlstm": ActionModelMyConvLSTM(),
}

results = []
for model_name, model in models_to_compare.items():
    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    result = run_experiment_task3(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=EPOCHS,
        lr=3e-4,
        log_dir="runs/kth_actions",
    )
    results.append(result)

comparison_df = pd.DataFrame(results).set_index("model")
comparison_df = comparison_df[
    [
        "params",
        "best_val_acc",
        "test_acc",
        "test_loss",
        "train_time_sec",
        "test_eval_sec",
        "inference_ms_per_sample",
    ]
]

display(comparison_df.round(4))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

comparison_df["test_acc"].plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Test accuracy")
axes[0].set_ylabel("accuracy")
axes[0].tick_params(axis="x", rotation=25)

comparison_df["params"].plot(kind="bar", ax=axes[1], color="darkorange")
axes[1].set_title("Learnable parameters")
axes[1].set_ylabel("# params")
axes[1].tick_params(axis="x", rotation=25)

comparison_df["train_time_sec"].plot(kind="bar", ax=axes[2], color="seagreen")
axes[2].set_title("Total training time")
axes[2].set_ylabel("seconds")
axes[2].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.show()