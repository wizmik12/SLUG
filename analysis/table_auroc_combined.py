"""Single LaTeX table: AUROC on the worst 5%, with SSIM and LPIPS as column groups.

Same numbers as tables/tail_auroc_ssim.tex and tail_auroc_lpips.tex, merged so the paper
carries one table instead of two. Ranking is per column.
"""
import subprocess, re, sys
from pathlib import Path

COLS = ["FST12", "FST56", "HUSC", "HCUV"]
rows = {}
order = []
for metric in ["SSIM", "LPIPS"]:
    out = subprocess.run([sys.executable, "analysis/tail_detection_multimetric.py",
                          "--metric", metric], capture_output=True, text=True).stdout
    body = out.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().splitlines()
    for line in body:
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        key = (cells[0], cells[1])
        if key not in rows:
            rows[key] = {}
            order.append(key)
        rows[key][metric] = cells[2:6]

L = [r"% AUROC, tail = worst 5% by SSIM (lowest) and by LPIPS (highest). Higher is better.",
     r"% Bold = best in column, underline = second. All methods scored against the",
     r"% DETERMINISTIC model's quality, since no per-method SSIM/LPIPS exists.",
     r"% Deep ensemble is a single measurement (the 5 seeds are its members), hence no std.",
     r"\begin{tabular}{ll cccc cccc}",
     r"\toprule",
     r" & & \multicolumn{4}{c}{SSIM} & \multicolumn{4}{c}{LPIPS} \\",
     r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
     r"Method & Memory & " + " & ".join(COLS) + " & " + " & ".join(COLS) + r" \\",
     r"\midrule"]
for name, mem in order:
    L.append(f"{name} & {mem} & " + " & ".join(rows[(name, mem)]["SSIM"])
             + " & " + " & ".join(rows[(name, mem)]["LPIPS"]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
Path("tables/tail_auroc_ssim_lpips.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))
print("\nwrote tables/tail_auroc_ssim_lpips.tex")
