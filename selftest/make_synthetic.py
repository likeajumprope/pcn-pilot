#!/usr/bin/env python3
"""Write a small synthetic multi-site cohort with the usual real-world mess.

    python make_synthetic.py <out_dir> [--n 1500] [--seed 7]

Two source tables whose IDs are formatted differently, numeric sex coding, a
diagnosis column, a PHI column, one sparse variable, one skewed variable, one
heteroskedastic variable, a unit error, a tiny site, and a new site held back
for transfer / extend tests. Nothing here is real data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def cohort(n: int, sites: dict, rng, start: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = list(sites)
    site = rng.choice(names, n, p=np.array([sites[s][0] for s in names]) / sum(sites[s][0] for s in names))
    lo = np.array([sites[s][1] for s in site])
    hi = np.array([sites[s][2] for s in site])
    age = rng.uniform(lo, hi)
    sex = rng.integers(0, 2, n)
    dx = rng.choice(["HC", "PAT"], n, p=[0.85, 0.15])
    off = np.array([sites[s][3] for s in site])
    a = (age - 45) / 20
    pat = (dx == "PAT") * 1.0
    ids = [f"{(i % 90) + 1:03d}_S_{start + i:04d}" for i in range(n)]
    demo = pd.DataFrame({"participant_id": ids, "age": np.round(age, 2), "sex": sex, "site": site,
                         "diagnosis": dx, "last_name": [f"Person{i}" for i in range(n)]})
    idp = pd.DataFrame({"SubjID": [s.replace("_", "") for s in ids]})
    for k in range(1, 5):
        idp[f"thick_{k}"] = 2.6 - 0.12 * a - 0.03 * k * a ** 2 + 0.08 * off + 0.03 * sex - 0.15 * pat + rng.normal(0, 0.12, n)
    idp["vol_skewed"] = np.exp(0.2 * a + 0.15 * off + 0.3 * pat + rng.normal(0, 0.45, n)) * 1000
    idp["vol_hetero"] = 5000 - 300 * a + 150 * off + rng.normal(0, 1, n) * (200 + 120 * (a + 2))
    idp["vol_site_var"] = 800 + 40 * a + rng.normal(0, 1, n) * np.where(off > 0.5, 120, 50)
    idp["sparse_measure"] = np.where(rng.random(n) < 0.4, np.nan, rng.normal(10, 1, n))
    return demo, idp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # site: (weight, age_lo, age_hi, offset)
    sites = {"SiteA": (30, 8, 80, 0.0), "SiteB": (25, 8, 80, 1.0), "SiteC": (20, 18, 80, -1.0),
             "SiteD": (15, 8, 60, 0.5), "SiteE": (10, 40, 80, -0.5), "Tiny": (0.5, 20, 60, 2.0)}
    demo, idp = cohort(a.n, sites, rng)
    idp.loc[5, "thick_1"] *= 1000                     # a unit error
    idp = idp.drop(index=[10, 11, 12]).sample(frac=1, random_state=1)   # three subjects lack imaging
    demo.to_csv(out / "demographics.csv", index=False)
    idp.to_csv(out / "idps.tsv", sep="\t", index=False)
    d2, i2 = cohort(260, {"NewSite": (1, 10, 75, 1.6)}, rng, start=5000)
    d2.to_csv(out / "newsite_demographics.csv", index=False)
    i2.to_csv(out / "newsite_idps.tsv", sep="\t", index=False)
    print(f"wrote 4 tables to {out}")


if __name__ == "__main__":
    main()
