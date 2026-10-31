"""Baseline comparison tables: correlation and AUROC, mean and standard deviation over the seeds.

Usage:
  python analysis/table_baselines_ranked.py --panel mse --out tables/baselines_mse.tex
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "fitz":    {"groups": [("light_skin", "FST12"), ("dark_skin", "FST56")]},
    "ai4skin": {"groups": [("HUSC", "HUSC"), ("HCUV", "HCUV")]},
}
# display name -> (file stem, memory label)
METHODS = [   # paper order: baselines, then the ensemble upper bound, ours last
    ("Encoder std.",        "uq_metrics_encoderstd",         r"$\sim\!0$"),
    ("MC-dropout",          "uq_metrics_dropout",            r"$\sim\!0$"),
    ("Low-rank LLA",        "uq_metrics_lowranklla_k1",      r"$1p$"),
    ("SWAG",                "uq_metrics_swag_nv1",           r"$3p$"),
    ("Deep ensemble (M=5)", "__ENSEMBLE__",                  r"$5p$"),
    ("SLUG (ours)",         "uq_metrics_slug_it100_sk10000", r"$0.018p$"),
]
PANEL_COL = {"mse": "MSE", "ssim": "SSIM", "lpips": "LPIPS"}


def load_metrics(root, ds, seed, group):
    p = Path(root) / ds / "Results" / f"0_seed{seed}" / "0.5" / f"metrics_{group}.csv"
    df = pd.read_csv(p).rename(columns={"Image Name": "image_id"})
    return df.set_index("image_id")


def load_uq(root, ds, seed, group, stem):
    p = Path(root) / ds / "Results" / f"0_seed{seed}" / "0.5" / f"{stem}_{group}.csv"
    df = pd.read_csv(p).rename(columns={"Image Name": "image_id", "Uncertainty": "score"})
    return df.set_index("image_id")["score"]


def load_ensemble(root, ds, group):
    p = Path(root) / ds / f"ensemble_var_{group}.csv"
    df = pd.read_csv(p).rename(columns={"ensemble_var": "score"})
    return df.set_index("image_id")["score"]


def corr(score, target):
    j = pd.concat([score.rename("s"), target.rename("t")], axis=1).dropna()
    return pearsonr(j["s"], j["t"])[0], spearmanr(j["s"], j["t"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="outputs/uq_baselines_performance")
    ap.add_argument("--panel", default="mse", choices=["mse", "ssim", "lpips"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--self_target", action="store_true",
                    help="correlate each method against the Error column in its OWN csv "
                         "(matters only for MC-dropout, whose stochastic forward pass "
                         "reconstructs differently from the deterministic model)")
    a = ap.parse_args()
    col = PANEL_COL[a.panel]
    higher_is_better = a.panel != "ssim"   # SSIM: want NEGATIVE correlation

    cols = []          # (dataset, group_label)
    cells = {}         # (method, col) -> (mean_p, std_p, mean_s, std_s)
    for ds, cfg in DATASETS.items():
        for group, label in cfg["groups"]:
            cols.append(label)
            targets = {s: load_metrics(a.results_dir, ds, s, group)[col] for s in SEEDS}
            # Common target for every method: the seed-averaged quality metric. With per-seed
            # targets each method is correlated against a different reference, so row-to-row
            # differences partly reflect the reference's sampling noise rather than the score.
            # Averaging removes that, and the +- then measures only how the METHOD varies
            # across seeds. --self_target overrides this for the MSE panel.
            avg_target = pd.concat(targets.values(), axis=1).mean(axis=1)
            for name, stem, mem in METHODS:
                if stem == "__ENSEMBLE__":
                    sc = load_ensemble(a.results_dir, ds, group)
                    p, s = corr(sc, avg_target)
                    cells[(name, mem, label)] = (p, np.nan, s, np.nan)
                else:
                    ps, ss = [], []
                    for seed in SEEDS:
                        try:
                            sc = load_uq(a.results_dir, ds, seed, group, stem)
                        except FileNotFoundError:
                            continue
                        tgt = avg_target
                        if a.self_target and a.panel == "mse":
                            f = Path(a.results_dir)/ds/"Results"/f"0_seed{seed}"/"0.5"/f"{stem}_{group}.csv"
                            own = pd.read_csv(f).rename(columns={"Image Name": "image_id"})
                            if "Error" in own.columns:
                                tgt = own.set_index("image_id")["Error"]
                        p, s = corr(sc, tgt)
                        ps.append(p); ss.append(s)
                    if not ps:
                        cells[(name, mem, label)] = (np.nan,)*4
                    else:
                        cells[(name, mem, label)] = (np.mean(ps), np.std(ps), np.mean(ss), np.std(ss))

    # rank per column, per statistic
    def ranked(label, idx):
        vals = {(n,mm): cells[(n, mm, label)][idx] for n, _, mm in METHODS}
        ok = {n: v for n, v in vals.items() if not np.isnan(v)}
        order = sorted(ok, key=lambda n: ok[n], reverse=higher_is_better)
        return (order[0] if order else None), (order[1] if len(order) > 1 else None)

    best = {(l, i): ranked(l, i) for l in cols for i in (0, 2)}

    def fmt(name, mem, label, idx):
        m, sd = cells[(name, mem, label)][idx], cells[(name, mem, label)][idx + 1]
        if np.isnan(m):
            return "--"
        txt = f"{m:.3f}" if np.isnan(sd) else f"{m:.3f}\\,$\\pm$\\,{sd:.3f}"
        b, sec = best[(label, idx)]
        if (name, mem) == b:
            return r"\textbf{" + txt + "}"
        if (name, mem) == sec:
            return r"\underline{" + txt + "}"
        return txt

    L = []
    L.append(r"% Auto-generated by analysis/table_baselines_ranked.py --panel " + a.panel)
    L.append(r"% Requires: \usepackage{booktabs} and (for \underline on numbers) nothing extra.")
    L.append(r"% Bold = best, underline = second best, ranked on the SIGNED correlation.")
    L.append(r"\begin{center}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{llcccc}")
    L.append(r"\toprule")
    L.append(r"Method & Memory & " + " & ".join(cols) + r" \\")
    L.append(r"\midrule")
    for name, _, mem in METHODS:
        L.append(f"{name} & {mem} & " + " & ".join(fmt(name, mem, l, 0) for l in cols) + r" \\")
        L.append(r"\quad\textit{Spearman} & & " + " & ".join(fmt(name, mem, l, 2) for l in cols) + r" \\")
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    L.append(r"\end{center}")
    tex = "\n".join(L)

    print(f"=== panel: {a.panel} ({'higher' if higher_is_better else 'more negative'} correlation is better) ===\n")
    hdr = f"{'method':16s} {'memory':9s} " + " ".join(f"{c:>18s}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for name, _, mem in METHODS:
        row = f"{name:16s} {mem.replace(chr(92),'').replace('$',''):9s} "
        for l in cols:
            m, sd = cells[(name, mem, l)][0], cells[(name, mem, l)][1]
            row += f"{m:>11.3f}+-{0 if np.isnan(sd) else sd:<5.3f}" if not np.isnan(m) else f"{'--':>18s}"
        print(row)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(tex + "\n")
        print(f"\nwrote {a.out}")
    else:
        print("\n" + tex)


if __name__ == "__main__":
    main()
