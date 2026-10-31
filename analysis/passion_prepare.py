"""Prepare PASSION for scoring: cache resized images and build a labelled manifest.
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import transforms

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/work/work_mik/2026/SLUG/PASSION_MICCAI_2024")
ap.add_argument("--out", default="/work/work_mik/2026/SLUG/PASSION_MICCAI_2024/prepared128")
ap.add_argument("--image_size", type=int, default=128)
ap.add_argument("--workers", type=int, default=16)
cli = ap.parse_args()

SRC = Path(cli.src); OUT = Path(cli.out); IMGDIR = OUT / "images_resized"
IMGDIR.mkdir(parents=True, exist_ok=True)

tf = transforms.Compose([transforms.Resize(cli.image_size),
                         transforms.CenterCrop(cli.image_size)])


def one(fn):
    dst = IMGDIR / fn
    if dst.exists():
        return fn, True
    try:
        im = Image.open(SRC / "images" / fn).convert("RGB")
        tf(im).save(dst, quality=95)
        return fn, True
    except Exception as e:
        print(f"  FAILED {fn}: {e}")
        return fn, False


files = sorted(os.listdir(SRC / "images"))
print(f"resizing {len(files)} images -> {IMGDIR} at {cli.image_size}px")
with ProcessPoolExecutor(cli.workers) as ex:
    res = list(ex.map(one, files, chunksize=16))
ok = [f for f, good in res if good]
print(f"done: {len(ok)}/{len(files)} written")

lab = pd.read_csv(SRC / "label.csv")
man = pd.DataFrame({"file": ok})
man["subject_id"] = man["file"].str.split("_").str[0]
man = man.merge(lab, on="subject_id", how="left")
missing = man["country"].isna().sum()
print(f"manifest: {len(man)} rows, {man.subject_id.nunique()} subjects, "
      f"{missing} without a label row")
man.to_csv(OUT / "manifest.csv", index=False)
print(f"wrote {OUT/'manifest.csv'}")
for c in ["country", "fitzpatrick", "sex", "conditions_PASSION"]:
    print(f"\n-- {c} (images) --\n{man[c].value_counts(dropna=False).head(8)}")
