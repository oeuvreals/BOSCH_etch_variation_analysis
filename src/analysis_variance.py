"""산포 계층 분해: lot 간 / 순번 drift / 웨이퍼 간 랜덤 / 웨이퍼 내 위치 / 잔차.

주의 — 측정 위치는 모든 웨이퍼에서 동일한 고정 좌표다. 위치 효과를 랜덤 잔차로
두는 통상적 nested ANOVA 를 쓰면 웨이퍼 평균의 산포가 전부 "9점 샘플링 노이즈"로
흡수되어 웨이퍼 간 성분이 0 으로 붕괴한다. 따라서 위치를 고정 인자로 분리한
직교 제곱합 분해를 쓴다.

    y_ijk = µ + lot_i + wafer_ij + loc_k + e_ijk

모든 웨이퍼가 동일 위치 집합을 가지므로 위 분해는 정확히 직교한다.

출력:
    data/processed/variance_components_{9,89}pt.csv
    figures/variance_decomposition.png
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
DATA_DIRS = [ROOT / "datasets", ROOT / "data" / "raw"]

VALUE = "si_etch"
LABELS = [
    "① lot 간 (날짜·조건)",
    "②a lot 내 순번 drift",
    "②b 웨이퍼 간 랜덤",
    "③ 웨이퍼 내 위치",
    "④ 잔차 (위치×웨이퍼)",
]


def find(name):
    for d in DATA_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def load_points(n_points):
    df = pd.read_csv(find(f"Si_Oxide_etch_{n_points}_points.csv")).dropna(subset=["experiment_key"])
    df[["lot_number", "wafer_number"]] = df[["lot_number", "wafer_number"]].astype(int)
    df["loc"] = df["X"].astype(str) + "," + df["Y"].astype(str)
    df["wafer_uid"] = df["lot_number"].astype(str) + "_" + df["wafer_number"].astype(str)
    return df


def decompose(df):
    y = df[VALUE]
    grand = y.mean()
    lot_mean = y.groupby(df["lot_number"]).transform("mean")
    wafer_mean = y.groupby(df["wafer_uid"]).transform("mean")
    loc_mean = y.groupby(df["loc"]).transform("mean")

    ss = {
        "lot": ((lot_mean - grand) ** 2).sum(),
        "wafer": ((wafer_mean - lot_mean) ** 2).sum(),
        "loc": ((loc_mean - grand) ** 2).sum(),
    }
    resid = y - wafer_mean - loc_mean + grand
    ss["resid"] = (resid**2).sum()
    ss_total = ((y - grand) ** 2).sum()

    gap = abs(sum(ss.values()) - ss_total) / ss_total
    assert gap < 1e-6, f"직교 분해 불성립 (오차 {gap:.2e}) — 위치 집합이 웨이퍼마다 다름"
    return ss, ss_total


def drift_fit(df):
    """lot 평균을 뺀 웨이퍼 평균에 대한 순번 선형추세."""
    wm = df.groupby(["lot_number", "wafer_number"])[VALUE].mean().reset_index()
    wm["centered"] = wm[VALUE] - wm.groupby("lot_number")[VALUE].transform("mean")
    fit = smf.ols("centered ~ wafer_number", data=wm).fit()
    return fit.rsquared, fit.params["wafer_number"], fit.pvalues["wafer_number"]


def crosscheck(df, ss):
    """OLS 로 동일 제곱합이 나오는지 확인.

    lot 은 wafer_uid 에 완전히 포함되므로 두 인자를 한 모델에 넣으면 rank deficient
    가 되어 anova_lm 의 제곱합이 깨진다. 인자별 단독 적합으로 비교한다.
    """
    ess = lambda f: smf.ols(f"{VALUE} ~ {f}", data=df).fit().ess
    got = {
        "lot": ess("C(lot_number)"),
        "lot+wafer": ess("C(wafer_uid)"),
        "loc": ess("C(loc)"),
        "explained": ess("C(wafer_uid) + C(loc)"),
    }
    want = {
        "lot": ss["lot"],
        "lot+wafer": ss["lot"] + ss["wafer"],
        "loc": ss["loc"],
        "explained": ss["lot"] + ss["wafer"] + ss["loc"],
    }
    return max(abs(got[k] - want[k]) for k in want)


def analyze(n_points):
    df = load_points(n_points)
    ss, ss_total = decompose(df)
    r2, slope, pval = drift_fit(df)

    parts = [ss["lot"], ss["wafer"] * r2, ss["wafer"] * (1 - r2), ss["loc"], ss["resid"]]
    point = pd.DataFrame({"component": LABELS, "sum_sq": parts})
    point["pct"] = 100 * point["sum_sq"] / ss_total
    point["sd_equiv_um"] = np.sqrt(point["sum_sq"] / len(df))

    # 웨이퍼 평균만 보는 관점 (위치·잔차 성분이 빠짐)
    n_wafers = df["wafer_uid"].nunique()
    ss_mean_total = ss["lot"] + ss["wafer"]
    wmean = pd.DataFrame(
        {
            "component": LABELS[:3],
            "sum_sq": [ss["lot"], ss["wafer"] * r2, ss["wafer"] * (1 - r2)],
        }
    )
    wmean["pct"] = 100 * wmean["sum_sq"] / ss_mean_total
    # 점 단위 SS = (웨이퍼당 측정점 수) x (웨이퍼 평균 편차 제곱합)
    n_per_wafer = len(df) / n_wafers
    wmean["sd_equiv_um"] = np.sqrt(wmean["sum_sq"] / n_per_wafer / n_wafers)

    print(f"\n=== {n_points}점 : lot {df['lot_number'].nunique()}개 / 웨이퍼 {n_wafers}장 / 측정점 {len(df)}개 ===")
    print("[관점 A] 측정점 전체 기준")
    print(point.round(4).to_string(index=False))
    print(f"전체 σ = {np.sqrt(ss_total / len(df)):.3f} µm")
    print("\n[관점 B] 웨이퍼 평균 기준")
    print(wmean.round(4).to_string(index=False))
    print(f"웨이퍼 평균의 σ = {np.sqrt(ss_mean_total / n_per_wafer / n_wafers):.3f} µm")
    print(f"\ndrift {slope:+.4f} µm/웨이퍼 (p={pval:.1e}) — 웨이퍼 간 변동의 {100 * r2:.1f}% 설명")
    print(f"OLS 교차검증 최대 절대오차 {crosscheck(df, ss):.2e} (총 SS {ss_total:.1f})")

    point.assign(view="point").to_csv(PROC / f"variance_components_{n_points}pt.csv", index=False)
    return point, wmean


COLORS = dict(zip(LABELS, ["#4C72B0", "#DD8452", "#C7C7C7", "#55A868", "#EDEDED"]))


def _stack(ax, out, title):
    bottom = 0.0
    for _, row in out.iterrows():
        c = COLORS[row["component"]]
        ax.bar(0, row["pct"], bottom=bottom, color=c, width=0.6, label=row["component"])
        if row["pct"] > 2.5:
            ax.text(0, bottom + row["pct"] / 2, f"{row['pct']:.1f}%", ha="center", va="center",
                    color="black" if c in ("#C7C7C7", "#EDEDED") else "white", fontweight="bold")
        bottom += row["pct"]
    ax.set_xticks([])
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10)


def plot(res):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))

    for i, (n, (point, wmean)) in enumerate(res.items()):
        _stack(axes[2 * i], point, f"{n}점 · 측정점 전체")
        _stack(axes[2 * i + 1], wmean, f"{n}점 · 웨이퍼 평균")
    axes[0].set_ylabel("제곱합 기여도 (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="center right", frameon=False, fontsize=9)
    fig.suptitle("Si 식각 깊이 산포의 계층별 분해", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.76, 1])
    fig.savefig(FIG / "variance_decomposition.png", dpi=150)
    print(f"\n-> {(FIG / 'variance_decomposition.png').relative_to(ROOT)}")


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plot({n: analyze(n) for n in (9, 89)})


if __name__ == "__main__":
    main()
