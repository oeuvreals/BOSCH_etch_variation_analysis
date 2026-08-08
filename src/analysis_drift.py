"""lot 내 웨이퍼 순번 drift: lot 별 기울기, 컨디셔닝 조건별 비교, 선택비 추세.

출력:
    data/processed/drift_slopes_{9,89}pt.csv
    figures/drift_trend.png
    figures/drift_slope_by_condition.png
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

COND_COLOR = {"Chuck": "#4C72B0", "Si": "#55A868", "SiO2": "#DD8452"}


def style():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def load(n_points):
    return pd.read_csv(PROC / f"wafers_{n_points}pt.csv")


def lot_slopes(w, value="si_etch_mean"):
    rows = []
    for lot, g in w.groupby("lot_number"):
        fit = smf.ols(f"{value} ~ wafer_number", data=g).fit()
        lo, hi = fit.conf_int().loc["wafer_number"]
        first = g.loc[g.wafer_number.idxmin(), value]
        rows.append(
            {
                "lot_number": lot,
                "cond_type": g["cond_type"].iloc[0],
                "cond_rep": g["cond_rep"].iloc[0],
                "cond_label": g["cond_label"].iloc[0],
                "date": g["date"].iloc[0],
                "n_wafers": len(g),
                "level_mean": g[value].mean(),
                "slope": fit.params["wafer_number"],
                "ci_lo": lo,
                "ci_hi": hi,
                "p_value": fit.pvalues["wafer_number"],
                "r2": fit.rsquared,
                "pct_per_10w": 100 * fit.params["wafer_number"] * 9 / first,
            }
        )
    return pd.DataFrame(rows)


def pooled_slope(w, value="si_etch_mean"):
    """lot 을 고정효과로 흡수한 공통 기울기."""
    fit = smf.ols(f"{value} ~ C(lot_number) + wafer_number", data=w).fit()
    lo, hi = fit.conf_int().loc["wafer_number"]
    return fit.params["wafer_number"], lo, hi, fit.pvalues["wafer_number"]


def condition_test(sl):
    """조건별 기울기 차이가 있는가 (lot 8개/10개뿐이라 검정력은 낮음)."""
    out = {}
    for factor in ("cond_type", "cond_rep"):
        if sl[factor].nunique() < 2:
            continue
        fit = smf.ols(f"slope ~ C({factor})", data=sl).fit()
        out[factor] = (fit.fvalue, fit.f_pvalue, sl.groupby(factor)["slope"].mean())
    return out


def report(n_points):
    w = load(n_points)
    sl = lot_slopes(w)
    slope, lo, hi, p = pooled_slope(w)

    print(f"\n=== {n_points}점 : lot 내 순번 drift ===")
    print(sl.round(4).to_string(index=False))
    print(f"\n공통 기울기 {slope:+.4f} µm/웨이퍼 [95% CI {lo:+.4f}, {hi:+.4f}] p={p:.1e}")
    print(f"10장 처리 시 누적 {slope * 9:+.3f} µm ({100 * slope * 9 / w.si_etch_mean.mean():+.2f}%)")

    print("\n선택비 추세:")
    ssl = lot_slopes(w, "selectivity")
    s_slope, s_lo, s_hi, s_p = pooled_slope(w, "selectivity")
    print(f"  공통 기울기 {s_slope:+.4f} /웨이퍼 [{s_lo:+.4f}, {s_hi:+.4f}] p={s_p:.1e}")
    o_slope, o_lo, o_hi, o_p = pooled_slope(w, "oxide_etch_mean")
    print(f"  oxide_etch {o_slope:+.5f} µm/웨이퍼 [{o_lo:+.5f}, {o_hi:+.5f}] p={o_p:.1e}")

    print("\n조건별 기울기 비교:")
    for factor, (f, pv, means) in condition_test(sl).items():
        print(f"  {factor}: F={f:.2f}, p={pv:.3f}")
        print("   ", means.round(4).to_dict())

    sl["value"] = "si_etch_mean"
    ssl["value"] = "selectivity"
    pd.concat([sl, ssl]).to_csv(PROC / f"drift_slopes_{n_points}pt.csv", index=False)
    return w, sl


def plot_trend(data):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (n, (w, sl)) in zip(axes, data.items()):
        for lot, g in w.groupby("lot_number"):
            g = g.sort_values("wafer_number")
            dev = g["si_etch_mean"] - g["si_etch_mean"].mean()
            c = COND_COLOR[g["cond_type"].iloc[0]]
            ax.plot(g["wafer_number"], dev, "o-", color=c, alpha=0.45, ms=4, lw=1)
        pooled = w.assign(dev=w["si_etch_mean"] - w.groupby("lot_number")["si_etch_mean"].transform("mean"))
        fit = smf.ols("dev ~ wafer_number", data=pooled).fit()
        x = np.arange(1, w["wafer_number"].max() + 1)
        ax.plot(x, fit.params["Intercept"] + fit.params["wafer_number"] * x, "k-", lw=2.5,
                label=f"공통 추세 {fit.params['wafer_number']:+.3f} µm/장")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel("lot 내 웨이퍼 순번")
        ax.set_title(f"{n}점 계측")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Si 식각 깊이 − lot 평균 (µm)")
    handles = [plt.Line2D([], [], color=c, marker="o", ls="-", label=f"컨디셔닝 {k}") for k, c in COND_COLOR.items()]
    axes[1].legend(handles=handles + axes[1].get_legend_handles_labels()[0], frameon=False, fontsize=9)
    fig.suptitle("lot 내 웨이퍼 순번에 따른 식각 깊이 감소 (drift)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "drift_trend.png", dpi=150)


def plot_slopes(data):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (n, (_, sl)) in zip(axes, data.items()):
        sl = sl.sort_values("lot_number")
        x = np.arange(len(sl))
        ax.bar(x, sl["slope"], color=[COND_COLOR[t] for t in sl["cond_type"]],
               yerr=[sl["slope"] - sl["ci_lo"], sl["ci_hi"] - sl["slope"]], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"lot {int(l)}\n{c}" for l, c in zip(sl["lot_number"], sl["cond_label"])], fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{n}점 계측")
    axes[0].set_ylabel("drift 기울기 (µm/웨이퍼)")
    fig.suptitle("lot 별 drift 기울기와 95% 신뢰구간", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "drift_slope_by_condition.png", dpi=150)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {n: report(n) for n in (9, 89)}
    plot_trend(data)
    plot_slopes(data)
    print("\n-> figures/drift_trend.png, figures/drift_slope_by_condition.png")


if __name__ == "__main__":
    main()
