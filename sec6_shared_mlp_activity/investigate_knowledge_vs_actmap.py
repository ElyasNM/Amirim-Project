#!/usr/bin/env python3
"""Does model knowledge / answer-extraction relate to cross-lingual MLP
activation-map similarity, or is the similarity fact-identity driven?

Trimmed: naive contrast -> relation control -> within-fact control -> typology
correction -> per-layer gap. Per-fact data is scope=mean (cosine_z, overlap_topk).
transfer = aligned - control.  bucket in {both, only_one, neither} per (fact,pair).
"""
import os, gzip
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import MLP_ROOT
ROOT = str(MLP_ROOT)
OUT = os.path.join(ROOT, "findings3_investigation"); os.makedirs(OUT, exist_ok=True)
RUNS = {
 "Llama last_subject": "results/meta-llama_Llama-3.1-8B_subj",
 "Llama last_token":   "results/meta-llama_Llama-3.1-8B",
 "Qwen last_token":    "results/Qwen_Qwen2.5-7B",
}
NK = {"neither": 0, "only_one": 1, "both": 2}
rng = np.random.default_rng(0)
L = []
def p(s=""):
    print(s); L.append(s)

def d_cohen(a, b):
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp if sp > 0 else np.nan

def within_fact_delta(m, col="transfer", kmin=2):
    out = []
    for idx, s in m.groupby("index"):
        b = s.loc[s.bucket == "both", col]; n = s.loc[s.bucket == "neither", col]
        if len(b) >= kmin and len(n) >= kmin:
            out.append(b.mean() - n.mean())
    return np.array(out)

def analyse(name, folder):
    p("\n" + "="*70); p(f"RUN: {name}"); p("="*70)
    with gzip.open(os.path.join(ROOT, folder, "per_fact_similarity.csv.gz"), "rt") as fh:
        d = pd.read_csv(fh)
    d["pair"] = d.lang1 + "-" + d.lang2
    d["transfer"] = d.aligned - d.control
    row = {}
    for metric in ["cosine_z", "overlap_topk"]:
        m = d[d.metric == metric].copy()
        both = m.loc[m.bucket == "both", "transfer"].to_numpy()
        only1 = m.loc[m.bucket == "only_one", "transfer"].to_numpy()
        neith = m.loc[m.bucket == "neither", "transfer"].to_numpy()
        p(f"\n-- {metric}  (n={len(m)} fact*pair) --")
        p(f"A. naive transfer:  both={both.mean():.4f}  only_one={only1.mean():.4f}  "
          f"neither={neith.mean():.4f}   both-neither={both.mean()-neith.mean():+.4f}  "
          f"d={d_cohen(both, neith):+.3f}")
        cb = m.loc[m.bucket == "both", "control"].mean(); cn = m.loc[m.bucket == "neither", "control"].mean()
        ab = m.loc[m.bucket == "both", "aligned"].mean(); an = m.loc[m.bucket == "neither", "aligned"].mean()
        p(f"   aligned: both={ab:.4f} neither={an:.4f} ({ab-an:+.4f}) | "
          f"control null: both={cb:+.4f} neither={cn:+.4f} ({cb-cn:+.4f})")

        rel = m.groupby("relation").agg(n=("transfer", "size"),
              fb=("bucket", lambda s: (s == "both").mean()),
              mt=("transfer", "mean"))
        p(f"B. across {len(rel)} relations: mean_transfer range {rel.mt.max()-rel.mt.min():.3f}, "
          f"frac_both range {rel.fb.max()-rel.fb.min():.2f}, "
          f"corr(frac_both, transfer)={np.corrcoef(rel.fb, rel.mt)[0,1]:+.3f}")

        dl, w = [], []
        for _, s in m.groupby("relation"):
            b = s.loc[s.bucket == "both", "transfer"]; n = s.loc[s.bucket == "neither", "transfer"]
            if len(b) >= 20 and len(n) >= 20:
                dl.append(b.mean()-n.mean()); w.append(min(len(b), len(n)))
        dl = np.array(dl); w = np.array(w)
        p(f"C. within-relation both-neither: wmean={np.average(dl, weights=w):+.4f}  "
          f"{(dl>0).sum()}/{len(dl)} positive  range[{dl.min():+.3f},{dl.max():+.3f}]")

        wf = within_fact_delta(m)
        _, pw = stats.wilcoxon(wf)
        boot = np.array([rng.choice(wf, len(wf), replace=True).mean() for _ in range(3000)])
        ci = np.percentile(boot, [2.5, 97.5])
        sg = stats.binomtest((wf > 0).sum(), len(wf), 0.5).pvalue
        p(f"D. within-FACT both-neither (n={len(wf)} facts): mean={wf.mean():+.4f} "
          f"CI[{ci[0]:+.4f},{ci[1]:+.4f}]  Wilcoxon p={pw:.1e}  sign {(wf>0).sum()}/{len(wf)} p={sg:.1e}")

        pb = m.groupby("pair")["transfer"].mean()
        m["transfer_adj"] = m.transfer - m.pair.map(pb)
        rows = []
        for idx, s in m.groupby("index"):
            b = s[s.bucket == "both"]; n = s[s.bucket == "neither"]
            if len(b) >= 2 and len(n) >= 2:
                rows.append((b.transfer_adj.mean()-n.transfer_adj.mean(),
                             b.pair.map(pb).mean()-n.pair.map(pb).mean()))
        adj = pd.DataFrame(rows, columns=["delta_adj", "typo_gap"])
        _, pa = stats.wilcoxon(adj.delta_adj)
        p(f"F. within-FACT after removing each pair's typology baseline: "
          f"mean={adj.delta_adj.mean():+.4f}  Wilcoxon p={pa:.1e}   "
          f"(both-pairs are {adj.typo_gap.mean():+.4f} higher-typology than neither-pairs)")

        hp = pb.sort_values().tail(len(pb)//4).index
        mh = m[m.pair.isin(hp)]
        bh = mh.loc[mh.bucket == "both", "transfer"].to_numpy()
        nh = mh.loc[mh.bucket == "neither", "transfer"].to_numpy()
        p(f"G. top-quartile pairs by transfer ({len(hp)} pairs): "
          f"both-neither={bh.mean()-nh.mean():+.4f}  d={d_cohen(bh, nh):+.3f}")

        row[metric] = dict(naive_bn=both.mean()-neith.mean(), naive_d=d_cohen(both, neith),
            byrel_bn=np.average(dl, weights=w), byfact_bn=wf.mean(), byfact_p=pw,
            typo_adj_bn=adj.delta_adj.mean(), typo_adj_p=pa,
            typo_gap=adj.typo_gap.mean(), hi_bn=bh.mean()-nh.mean())
    return d, row

def per_layer(name, folder):
    d = pd.read_csv(os.path.join(ROOT, folder, "pairwise_transfer.csv"))
    d = d[(d.metric == "cosine_z") & d.scope.str.startswith("layer")]
    g = d.groupby(["scope", "bucket"])["transfer_score"].mean().unstack()
    g["gap"] = g["both"] - g["neither"]
    g["Ln"] = [int(s[5:]) for s in g.index]
    g = g.sort_values("Ln")
    p(f"per-layer {name}: transfer peaks L{g.loc[g['both'].idxmax(),'Ln']}, "
      f"gap peaks L{g.loc[g['gap'].idxmax(),'Ln']}, max gap {g['gap'].max():+.4f}, "
      f"corr(transfer,gap) across layers {np.corrcoef(g['both'], g['gap'])[0,1]:+.3f}")
    return g

def main():
    S, D = {}, {}
    for n, f in RUNS.items():
        D[n], S[n] = analyse(n, f)
    p("\n" + "="*70); p("PER-LAYER KNOWLEDGE GAP (aggregated over facts)"); p("="*70)
    G = {n: per_layer(n, f) for n, f in RUNS.items()}

    p("\n" + "="*70); p("HEADLINE  cosine_z  (transfer = aligned - control)"); p("="*70)
    p(f"{'run':<19}{'naive b-n':>11}{'d':>7}{'by-relat':>10}{'by-fact':>10}"
      f"{'fact p':>9}{'+typo-adj':>11}{'adj p':>9}")
    for n, r in S.items():
        c = r["cosine_z"]
        p(f"{n:<19}{c['naive_bn']:>+11.4f}{c['naive_d']:>+7.2f}{c['byrel_bn']:>+10.4f}"
          f"{c['byfact_bn']:>+10.4f}{c['byfact_p']:>9.1e}{c['typo_adj_bn']:>+11.4f}{c['typo_adj_p']:>9.1e}")

    # fig 1
    m = D["Llama last_token"]; m = m[m.metric == "cosine_z"]
    piv = m.pivot_table(index="relation", columns="bucket", values="transfer", aggfunc="mean")
    piv = piv.reindex(columns=["neither", "only_one", "both"]).sort_values("both")
    ax = piv.plot(kind="barh", figsize=(8, 7))
    ax.set_xlabel("mean transfer (cosine_z)"); ax.set_title("Llama last_token: transfer by relation x bucket")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_bucket_by_relation.png"), dpi=110); plt.close()

    # fig 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (n, dd) in zip(axes, D.items()):
        wf = within_fact_delta(dd[dd.metric == "cosine_z"])
        ax.hist(wf, bins=40, color="#4C78A8"); ax.axvline(0, color="k", lw=1)
        ax.axvline(wf.mean(), color="crimson", lw=2, label=f"mean {wf.mean():+.3f}\nn={len(wf)}")
        ax.set_title(n); ax.set_xlabel("within-fact (both - neither)"); ax.legend(fontsize=8)
    axes[0].set_ylabel("facts")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_within_fact_delta.png"), dpi=110); plt.close()

    # fig 3
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (n, g) in zip(axes, G.items()):
        ax.plot(g.Ln, g["both"], "-o", color="#4C78A8", label="transfer (both)")
        a2 = ax.twinx(); a2.plot(g.Ln, g["gap"], "-s", color="crimson", ms=4, label="both-neither")
        a2.set_ylim(-0.01, max(0.08, g["gap"].max()*1.3))
        ax.set_title(n); ax.set_xlabel("layer"); ax.set_ylabel("transfer_score"); a2.set_ylabel("both-neither")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_layer_gap.png"), dpi=110); plt.close()

    rows = [dict(run=n, metric=mt, **c) for n, r in S.items() for mt, c in r.items()]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables.csv"), index=False)
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("# Knowledge vs activation-map similarity — investigation\n\n```\n" + "\n".join(L) + "\n```\n")
    p(f"\nwrote {OUT}/  (report.md, tables.csv, 3 figs)")

if __name__ == "__main__":
    main()
