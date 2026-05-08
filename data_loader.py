import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


BASE_DIR = Path(r"C:\Users\dafin\OneDrive\Desktop\ML_Dafina_AI")
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"


def find_subdir(root, keywords):
    """Find a subfolder whose name contains any of the keywords."""
    for p in root.iterdir():
        if p.is_dir() and any(kw in p.name.lower() for kw in keywords):
            return p
    raise FileNotFoundError(
        f"No subfolder matching {keywords} in {root}. "
        f"Found: {[p.name for p in root.iterdir() if p.is_dir()]}"
    )


def collect_pairs(root):
    """Return list of (image_path, mask_path) tuples for one DUTS folder."""
    root = Path(root)
    img_dir = find_subdir(root, ["image", "imgs"])
    msk_dir = find_subdir(root, ["mask", "gt"])
    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")))

    pairs = []
    for img_path in image_files:
        mask_path = msk_dir / (img_path.stem + ".png")
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))
    print(f"[{root.name}] {len(pairs)} pairs")
    return pairs


def make_splits(pairs, train_frac=0.70, val_frac=0.15, seed=42):
    """Deterministic 70/15/15 split."""
    rng = random.Random(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]


def save_split(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for img, msk in pairs:
            f.write(f"{img}\t{msk}\n")


def load_split(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                img, msk = line.split("\t")
                pairs.append((img, msk))
    return pairs


class DUTSDataset(Dataset):
    """
    DUTS Salient Object Detection dataset.

    Args:
        pairs:       list of (image_path, mask_path) tuples
        image_size:  spatial size (default 128)
        augment:     apply training augmentations
        aug_profile: 'basic' (flip + crop + brightness) or
                     'strong' (also: vertical flip, rotation, contrast, saturation)
    """

    def __init__(self, pairs, image_size=128, augment=False, aug_profile="basic"):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        assert aug_profile in ("basic", "strong")
        self.aug_profile = aug_profile

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")

        img = TF.resize(img, [self.image_size, self.image_size],
                        interpolation=TF.InterpolationMode.BILINEAR)
        msk = TF.resize(msk, [self.image_size, self.image_size],
                        interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img); msk = TF.hflip(msk)

            if random.random() < 0.5:
                crop_size = random.randint(int(self.image_size * 0.7), self.image_size)
                top = random.randint(0, self.image_size - crop_size)
                left = random.randint(0, self.image_size - crop_size)
                img = TF.crop(img, top, left, crop_size, crop_size)
                msk = TF.crop(msk, top, left, crop_size, crop_size)
                img = TF.resize(img, [self.image_size, self.image_size],
                                interpolation=TF.InterpolationMode.BILINEAR)
                msk = TF.resize(msk, [self.image_size, self.image_size],
                                interpolation=TF.InterpolationMode.NEAREST)

            if random.random() < 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))

            if self.aug_profile == "strong":
                if random.random() < 0.2:
                    img = TF.vflip(img); msk = TF.vflip(msk)
                if random.random() < 0.5:
                    angle = random.uniform(-15, 15)
                    img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                    msk = TF.rotate(msk, angle, interpolation=TF.InterpolationMode.NEAREST)
                if random.random() < 0.5:
                    img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
                if random.random() < 0.5:
                    img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))

        img = TF.to_tensor(img)
        msk = TF.to_tensor(msk)
        msk = (msk > 0.5).float()  
        return img, msk


if __name__ == "__main__":
    tr_root = DATA_DIR / "DUTS-TR"
    te_root = DATA_DIR / "DUTS-TE"
    pairs = collect_pairs(tr_root) + collect_pairs(te_root)
    print(f"\ntotal pairs: {len(pairs)}")

    train, val, test = make_splits(pairs, seed=42)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    save_split(train, SPLITS_DIR / "train.txt")
    save_split(val,   SPLITS_DIR / "val.txt")
    save_split(test,  SPLITS_DIR / "test.txt")
    print(f"\nsaved splits to {SPLITS_DIR}")

    ds = DUTSDataset(train, image_size=128, augment=True)
    img, msk = ds[0]
    print(f"\nsanity check: img={tuple(img.shape)} mask={tuple(msk.shape)} "
          f"img range=[{img.min():.3f},{img.max():.3f}] "
          f"mask values={torch.unique(msk).tolist()}")