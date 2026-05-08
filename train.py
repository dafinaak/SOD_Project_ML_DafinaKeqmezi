import sys
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from data_loader import DUTSDataset, load_split, SPLITS_DIR
from sod_model import SODNet


BASE_DIR = Path(r"C:\Users\dafin\OneDrive\Desktop\ML_Dafina_AI")
CKPT_DIR = Path(r"C:\Users\dafin\ml_checkpoints")
LOG_DIR  = BASE_DIR / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5
EPS = 1e-7
SEED = 42


class BCEIoULoss(nn.Module):
    """Loss = BCE(logits, target) + iou_weight * (1 - soft IoU)."""
    def __init__(self, iou_weight=0.5, smooth=1e-6):
        super().__init__()
        self.iou_weight = iou_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targ = target.view(target.size(0), -1)
        intersection = (probs * targ).sum(dim=1)
        union = probs.sum(dim=1) + targ.sum(dim=1) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1.0 - iou.mean()
        return bce_loss + self.iou_weight * iou_loss


def batch_metrics(pred_b, targ_b):
    """Compute per-image IoU/Precision/Recall/F1, then mean over the batch."""
    pred = pred_b.view(pred_b.size(0), -1).float()
    targ = targ_b.view(targ_b.size(0), -1).float()
    tp = (pred * targ).sum(1)
    fp = (pred * (1 - targ)).sum(1)
    fn = ((1 - pred) * targ).sum(1)
    iou  = tp / (tp + fp + fn + EPS)
    prec = tp / (tp + fp + EPS)
    rec  = tp / (tp + fn + EPS)
    f1   = 2 * prec * rec / (prec + rec + EPS)
    return iou.mean(), prec.mean(), rec.mean(), f1.mean()


EXPERIMENTS = {
    "baseline":             {"depth": 4, "aug_profile": "basic",  "iou_weight": 0.5, "lr": 1e-3, "epochs": 20, "batch_size": 16},
    "exp1_strong_aug":      {"depth": 4, "aug_profile": "strong", "iou_weight": 1.0, "lr": 1e-3, "epochs": 20, "batch_size": 16},
    "exp2_deeper_lower_lr": {"depth": 5, "aug_profile": "basic",  "iou_weight": 0.5, "lr": 5e-4, "epochs": 20, "batch_size": 8},
}
PATIENCE = 4

def set_seeds(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, epoch, best_val, epochs_no_improve, log_lines):
    """Save full training state to disk for resume."""
    torch.save({
        "model_state":       model.state_dict(),
        "optimizer_state":   optimizer.state_dict(),
        "epoch":             epoch,
        "best_val":          best_val,
        "epochs_no_improve": epochs_no_improve,
        "log_lines":         log_lines,
        "torch_rng_state":   torch.get_rng_state(),
        "python_rng_state":  random.getstate(),
        "numpy_rng_state":   np.random.get_state(),
    }, path)


def load_checkpoint(path, model, optimizer):
    if not path.exists():
        return 1, float("inf"), 0, []

    ckpt = torch.load(path, map_location=DEVICE)

    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        try:
            model.load_state_dict(ckpt)
            print(f"  loaded old-format weights from {path.name} (no resume state, "
                  f"starting fresh from epoch 1)", flush=True)
        except Exception as e:
            print(f"  warning: could not load old checkpoint {path.name}: {e}\n"
                  f"  starting completely fresh", flush=True)
        return 1, float("inf"), 0, []

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"])
    if "python_rng_state" in ckpt:
        random.setstate(ckpt["python_rng_state"])
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])

    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt["best_val"]
    epochs_no_improve = ckpt["epochs_no_improve"]
    log_lines = ckpt.get("log_lines", [])

    print(f"  RESUMED from {path.name}: last completed epoch {ckpt['epoch']} → "
          f"continuing at epoch {start_epoch}", flush=True)
    print(f"  best val loss so far: {best_val:.4f}, "
          f"epochs without improvement: {epochs_no_improve}", flush=True)
    return start_epoch, best_val, epochs_no_improve, log_lines


def run_epoch(model, loader, criterion, optimizer=None, epoch_num=None, log=None):
    is_train = optimizer is not None
    model.train(is_train)

    sums = {"loss": 0.0, "iou": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "mae": 0.0}
    n_samples = 0
    n_batches = len(loader)
    start = time.time()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for i, (imgs, masks) in enumerate(loader, 1):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(imgs, return_logits=True)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred_b = (probs > THRESHOLD).float()
                targ_b = (masks > THRESHOLD).float()
                iou_t, prec_t, rec_t, f1_t = batch_metrics(pred_b, targ_b)
                mae_t = (probs - masks).abs().mean()

            bs = imgs.size(0)
            sums["loss"] += loss.item()   * bs
            sums["iou"]  += iou_t.item()  * bs
            sums["prec"] += prec_t.item() * bs
            sums["rec"]  += rec_t.item()  * bs
            sums["f1"]   += f1_t.item()   * bs
            sums["mae"]  += mae_t.item()  * bs
            n_samples += bs

            if is_train and (i % 50 == 0 or i == n_batches):
                avg_loss = sums["loss"] / n_samples
                avg_iou  = sums["iou"]  / n_samples
                msg = (f"  [epoch {epoch_num}] batch {i}/{n_batches} "
                       f"({100*i/n_batches:5.1f}%)  "
                       f"loss={avg_loss:.4f}  iou={avg_iou:.4f}  "
                       f"elapsed={(time.time()-start)/60:.1f}min")
                print(msg, flush=True)
                if log is not None:
                    log.append(msg)

    return {k: v / n_samples for k, v in sums.items()}


def main(name="baseline", fresh_start=False):
    if name not in EXPERIMENTS:
        print(f"unknown experiment '{name}'. choices: {list(EXPERIMENTS.keys())}", flush=True)
        sys.exit(1)

    set_seeds(SEED)

    cfg = EXPERIMENTS[name]
    ckpt_path = CKPT_DIR / f"{name}_best.pt"
    log_path = LOG_DIR / f"{name}.txt"

    print(f"=== Experiment: {name} ===", flush=True)
    print(f"config: {cfg}", flush=True)
    print(f"device: {DEVICE}", flush=True)
    print(f"checkpoint: {ckpt_path}", flush=True)
    print("", flush=True)

    print("loading splits...", flush=True)
    train_pairs = load_split(SPLITS_DIR / "train.txt")
    val_pairs   = load_split(SPLITS_DIR / "val.txt")
    print(f"train={len(train_pairs)}  val={len(val_pairs)}", flush=True)

    print("creating datasets...", flush=True)
    train_ds = DUTSDataset(train_pairs, 128, augment=True, aug_profile=cfg["aug_profile"])
    val_ds   = DUTSDataset(val_pairs,   128, augment=False)

    print("creating dataloaders...", flush=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    print("creating model...", flush=True)
    model = SODNet(depth=cfg["depth"]).to(DEVICE)
    criterion = BCEIoULoss(iou_weight=cfg["iou_weight"])
    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)

    if fresh_start and ckpt_path.exists():
        print("--fresh flag set, ignoring existing checkpoint", flush=True)
        start_epoch, best_val, epochs_no_improve, log_lines = 1, float("inf"), 0, []
    else:
        print("checking for existing checkpoint...", flush=True)
        start_epoch, best_val, epochs_no_improve, log_lines = load_checkpoint(
            ckpt_path, model, optimizer
        )

    if start_epoch == 1:
        log_lines = [
            f"=== Experiment: {name} ===",
            f"config: {cfg}",
            f"device: {DEVICE}",
            f"checkpoint: {ckpt_path}",
            f"train={len(train_pairs)}  val={len(val_pairs)}",
            f"params: {n_params:,}",
        ]
    else:
        log_lines.append(f"\n=== RESUMED at epoch {start_epoch} ===\n")

    print("", flush=True)

    overall_start = time.time()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, epoch, log_lines)
        val_metrics   = run_epoch(model, val_loader,   criterion, optimizer=None)

        epoch_mins = (time.time() - epoch_start) / 60
        improved = val_metrics["loss"] < best_val
        marker = "  <- best" if improved else ""

        epoch_msg = (
            f"\n=== epoch {epoch}/{cfg['epochs']}  ({epoch_mins:.1f}min){marker} ===\n"
            f"  TRAIN  loss={train_metrics['loss']:.4f}  "
            f"iou={train_metrics['iou']:.4f}  prec={train_metrics['prec']:.4f}  "
            f"rec={train_metrics['rec']:.4f}  f1={train_metrics['f1']:.4f}  "
            f"mae={train_metrics['mae']:.4f}\n"
            f"  VAL    loss={val_metrics['loss']:.4f}  "
            f"iou={val_metrics['iou']:.4f}  prec={val_metrics['prec']:.4f}  "
            f"rec={val_metrics['rec']:.4f}  f1={val_metrics['f1']:.4f}  "
            f"mae={val_metrics['mae']:.4f}\n"
        )
        print(epoch_msg, flush=True)
        log_lines.append(epoch_msg)

        compact = (f"=== epoch {epoch}/{cfg['epochs']}  "
                   f"train={train_metrics['loss']:.4f}  "
                   f"val={val_metrics['loss']:.4f}  "
                   f"({epoch_mins:.1f}min){marker} ===")
        log_lines.append(compact)

        if improved:
            best_val = val_metrics["loss"]
            epochs_no_improve = 0
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val,
                            epochs_no_improve, log_lines)
            print(f"  CHECKPOINT SAVED at epoch {epoch} (val_loss={best_val:.4f})",
                  flush=True)
        else:
            epochs_no_improve += 1
            resume_path = ckpt_path.with_suffix(".resume.pt")
            save_checkpoint(resume_path, model, optimizer, epoch, best_val,
                            epochs_no_improve, log_lines)
            if epochs_no_improve >= PATIENCE:
                msg = f"early stop after {PATIENCE} epochs without improvement"
                print(msg, flush=True)
                log_lines.append(msg)
                break

    total_msg = f"total time:    {(time.time()-overall_start)/60:.1f} min"
    best_msg  = f"best val loss: {best_val:.4f}"
    print(total_msg, flush=True)
    print(best_msg, flush=True)
    log_lines.append(total_msg)
    log_lines.append(best_msg)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nlog saved to {log_path}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    fresh = "--fresh" in args
    args = [a for a in args if a != "--fresh"]
    name = args[0] if args else "baseline"
    main(name, fresh_start=fresh)