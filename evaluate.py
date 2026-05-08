import sys
import csv
import random
from pathlib import Path
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data_loader import DUTSDataset, load_split, SPLITS_DIR
from sod_model import SODNet


BASE_DIR = Path(r"C:\Users\dafin\OneDrive\Desktop\ML_Dafina_AI")
CKPT_DIR = Path(r"C:\Users\dafin\ml_checkpoints")
VIZ_DIR  = BASE_DIR / "visualizations"
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS = {
    "baseline":             ("baseline_best.pt",            4),
    "exp1_strong_aug":      ("exp1_strong_aug_best.pt",     4),
    "exp2_deeper_lower_lr": ("exp2_deeper_lower_lr_best.pt", 5),
}
LEGACY_BASELINE = CKPT_DIR / "best_model.pt"


def compute_metrics(pred_b, targ_b, eps=1e-7):
    """IoU, Precision, Recall, F1 per sample. Inputs are binary tensors."""
    pred = pred_b.view(pred_b.size(0), -1).float()
    targ = targ_b.view(targ_b.size(0), -1).float()
    tp = (pred * targ).sum(1)
    fp = (pred * (1 - targ)).sum(1)
    fn = ((1 - pred) * targ).sum(1)
    iou = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return iou, prec, rec, f1


def compute_mae(probs, target):
    p = probs.view(probs.size(0), -1)
    t = target.view(target.size(0), -1)
    return torch.abs(p - t).mean(1)


def load_model(exp_name="baseline"):
    if exp_name not in RUNS:
        raise ValueError(f"unknown experiment '{exp_name}'. choices: {list(RUNS.keys())}")
    ckpt_file, depth = RUNS[exp_name]
    ckpt = CKPT_DIR / ckpt_file
    if not ckpt.exists() and exp_name == "baseline" and LEGACY_BASELINE.exists():
        ckpt = LEGACY_BASELINE  
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    model = SODNet(depth=depth).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f"loaded {ckpt}")
    return model


@torch.no_grad()
def metrics_on_test(exp_name="baseline"):
    model = load_model(exp_name)
    test_pairs = load_split(SPLITS_DIR / "test.txt")
    test_ds = DUTSDataset(test_pairs, 128, augment=False)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"test images: {len(test_pairs)}")

    iou_l, prec_l, rec_l, f1_l, mae_l = [], [], [], [], []
    for i, (imgs, masks) in enumerate(loader, 1):
        imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
        probs = model(imgs)
        pred_b = (probs > THRESHOLD).float()
        targ_b = (masks > THRESHOLD).float()
        iou, prec, rec, f1 = compute_metrics(pred_b, targ_b)
        mae = compute_mae(probs, masks)
        iou_l.append(iou.cpu()); prec_l.append(prec.cpu())
        rec_l.append(rec.cpu()); f1_l.append(f1.cpu()); mae_l.append(mae.cpu())
        if i % 20 == 0 or i == len(loader):
            print(f"  batch {i}/{len(loader)}")

    res = {
        "IoU":       torch.cat(iou_l).mean().item(),
        "Precision": torch.cat(prec_l).mean().item(),
        "Recall":    torch.cat(rec_l).mean().item(),
        "F1":        torch.cat(f1_l).mean().item(),
        "MAE":       torch.cat(mae_l).mean().item(),
    }
    print("\n" + "=" * 50)
    print(f"  TEST METRICS — {exp_name}")
    print("=" * 50)
    for k, v in res.items():
        print(f"  {k:<10}: {v:.4f}")
    print("=" * 50)

    out = BASE_DIR / f"test_metrics_{exp_name}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Test metrics — {exp_name}\n")
        for k, v in res.items():
            f.write(f"{k:<10}: {v:.4f}\n")
    print(f"saved {out}")
    return res


def to_np_img(t):  return t.cpu().permute(1, 2, 0).numpy()
def to_np_msk(t):  return t.cpu().squeeze(0).numpy()


def make_overlay(img_np, msk_np, color=(1.0, 0.0, 0.0), alpha=0.5):
    over = img_np.copy()
    color_layer = np.zeros_like(over)
    for c in range(3): color_layer[..., c] = color[c]
    m3 = np.repeat(msk_np[..., None], 3, axis=2)
    return np.clip(np.where(m3 > 0.5, (1 - alpha) * over + alpha * color_layer, over), 0, 1)


@torch.no_grad()
def visualize(exp_name="baseline", n_samples=12, seed=42):
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(exp_name)

    test_pairs = load_split(SPLITS_DIR / "test.txt")
    rng = random.Random(seed)
    sample_pairs = rng.sample(test_pairs, min(n_samples, len(test_pairs)))
    ds = DUTSDataset(sample_pairs, 128, augment=False)

    for i in range(len(ds)):
        img, mask = ds[i]
        prob = model(img.unsqueeze(0).to(DEVICE))[0]
        pred = (prob > THRESHOLD).float()
        img_np, gt_np, pred_np = to_np_img(img), to_np_msk(mask), to_np_msk(pred)
        overlay = make_overlay(img_np, pred_np)

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(img_np); ax[0].set_title("Input")
        ax[1].imshow(gt_np, cmap="gray"); ax[1].set_title("Ground Truth")
        ax[2].imshow(pred_np, cmap="gray"); ax[2].set_title("Prediction")
        ax[3].imshow(overlay); ax[3].set_title("Overlay")
        for a in ax: a.axis("off")
        plt.tight_layout()
        out = VIZ_DIR / f"{exp_name}_sample_{i:02d}.png"
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out.name}")

    n_rows = len(ds)
    fig, ax = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1: ax = ax[None, :]
    titles = ["Input", "Ground Truth", "Prediction", "Overlay"]
    for i in range(n_rows):
        img, mask = ds[i]
        prob = model(img.unsqueeze(0).to(DEVICE))[0]
        pred = (prob > THRESHOLD).float()
        img_np, gt_np, pred_np = to_np_img(img), to_np_msk(mask), to_np_msk(pred)
        overlay = make_overlay(img_np, pred_np)
        ax[i, 0].imshow(img_np)
        ax[i, 1].imshow(gt_np, cmap="gray")
        ax[i, 2].imshow(pred_np, cmap="gray")
        ax[i, 3].imshow(overlay)
        for j in range(4):
            ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(titles[j], fontsize=14)
    plt.tight_layout()
    grid_out = VIZ_DIR / f"{exp_name}_summary_grid.png"
    plt.savefig(grid_out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {grid_out.name}")


def compare_all():
    results = {}
    for name in RUNS:
        try:
            print(f"\nevaluating {name}...")
            results[name] = metrics_on_test(name)
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")

    if not results:
        print("no experiments found to compare")
        return

    print("\n" + "=" * 75)
    print(f"{'Experiment':<25} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'MAE':>8}")
    print("=" * 75)
    for n, m in results.items():
        print(f"{n:<25} {m['IoU']:>8.4f} {m['Precision']:>10.4f} "
              f"{m['Recall']:>8.4f} {m['F1']:>8.4f} {m['MAE']:>8.4f}")
    print("=" * 75)

    txt = BASE_DIR / "comparison_results.txt"
    csv_p = BASE_DIR / "comparison_results.csv"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"{'Experiment':<25} {'IoU':>8} {'Precision':>10} "
                f"{'Recall':>8} {'F1':>8} {'MAE':>8}\n")
        for n, m in results.items():
            f.write(f"{n:<25} {m['IoU']:>8.4f} {m['Precision']:>10.4f} "
                    f"{m['Recall']:>8.4f} {m['F1']:>8.4f} {m['MAE']:>8.4f}\n")
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Experiment", "IoU", "Precision", "Recall", "F1", "MAE"])
        for n, m in results.items():
            w.writerow([n, m["IoU"], m["Precision"], m["Recall"], m["F1"], m["MAE"]])
    print(f"\nsaved {txt}\nsaved {csv_p}")


@torch.no_grad()
def verify_with_sklearn(exp_name="baseline", n_batches=2):
    """
    Compute metrics on a few batches using both our manual PyTorch
    implementation AND sklearn, then print both side by side.

    This confirms our metric formulas match scikit-learn's standard
    implementations. Small differences are expected because our manual
    implementation uses per-image-then-averaged (standard for SOD literature),
    while sklearn computes global aggregation by default.
    """
    from sklearn.metrics import (
        jaccard_score, precision_score, recall_score,
        f1_score, mean_absolute_error
    )

    model = load_model(exp_name)
    test_pairs = load_split(SPLITS_DIR / "test.txt")
    test_ds = DUTSDataset(test_pairs, 128, augment=False)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    print(f"\n{'='*65}")
    print(f"  sklearn verification on '{exp_name}' (first {n_batches} batches)")
    print(f"{'='*65}")

    for i, (imgs, masks) in enumerate(loader, 1):
        if i > n_batches:
            break
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        probs = model(imgs)
        pred_b = (probs > THRESHOLD).float()
        targ_b = (masks > THRESHOLD).float()

        iou_ours, prec_ours, rec_ours, f1_ours = compute_metrics(pred_b, targ_b)
        mae_ours = compute_mae(probs, masks)

        p_flat   = pred_b.flatten().cpu().numpy().astype(int)
        t_flat   = targ_b.flatten().cpu().numpy().astype(int)
        prob_flt = probs.flatten().cpu().numpy()
        targ_flt = masks.flatten().cpu().numpy()

        iou_sk  = jaccard_score(t_flat, p_flat, zero_division=0)
        prec_sk = precision_score(t_flat, p_flat, zero_division=0)
        rec_sk  = recall_score(t_flat, p_flat, zero_division=0)
        f1_sk   = f1_score(t_flat, p_flat, zero_division=0)
        mae_sk  = mean_absolute_error(targ_flt, prob_flt)

        print(f"\n  batch {i}:")
        print(f"    IoU       manual: {iou_ours.mean().item():.4f}   sklearn: {iou_sk:.4f}")
        print(f"    Precision manual: {prec_ours.mean().item():.4f}   sklearn: {prec_sk:.4f}")
        print(f"    Recall    manual: {rec_ours.mean().item():.4f}   sklearn: {rec_sk:.4f}")
        print(f"    F1        manual: {f1_ours.mean().item():.4f}   sklearn: {f1_sk:.4f}")
        print(f"    MAE       manual: {mae_ours.mean().item():.4f}   sklearn: {mae_sk:.4f}")

    print(f"\n{'='*65}")
    print("  Note: small differences are expected.")
    print("  - Manual: per-image, then mean (SOD literature convention)")
    print("  - sklearn: global aggregation across all pixels")
    print("  Both are mathematically valid; metric formulas are identical.")
    print(f"{'='*65}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage:")
        print("  python evaluate.py metrics [exp_name]")
        print("  python evaluate.py visualize [exp_name]")
        print("  python evaluate.py compare")
        print("  python evaluate.py verify_sklearn [exp_name]")
        sys.exit(1)

    cmd = sys.argv[1]
    exp = sys.argv[2] if len(sys.argv) > 2 else "baseline"

    if cmd == "metrics":          metrics_on_test(exp)
    elif cmd == "visualize":      visualize(exp)
    elif cmd == "compare":        compare_all()
    elif cmd == "verify_sklearn": verify_with_sklearn(exp)
    else:
        print(f"unknown command '{cmd}'")
        sys.exit(1)