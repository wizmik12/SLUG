"""Prepare PASSION for scoring: cache resized images and build a labelled manifest.

Two jobs the existing code path does not do:

1. RESIZE. The raw images run up to 5250x6762; decoding those per batch dominates
   scoring time. We pre-apply exactly the transform src/data.py uses for the test
   split -- Resize(short side -> image_size) then CenterCrop(image_size) -- and cache
   the result, so the scoring run is pure compute.

2. JOIN THE LABELS. src/data.py's passion_dataset takes the dataframe and never uses
   it: it lists the image directory and returns (image, filename), so no subgroup ever
   reaches the scorer. Filenames are <subject_id>_<n>.jpg, which joins to label.csv on
   subject_id and gives country / fitzpatrick / sex / condition per image.

Output: <out>/images_resized/*.jpg  and  <out>/manifest.csv
(the directory is named images_resized/ so the existing loader path also works.)
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
