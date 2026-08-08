"""4단계 — lot 내 웨이퍼 순번 drift 정량화 + 컨디셔닝 조건별 비교.

3단계에서 웨이퍼 평균 산포의 62~70% 가 "lot 내 순번 drift" 임을 밝혔다.
여기서는 그 drift 를 기울기(µm/장)로 정량화하고, DOE 의 존재 이유인
"컨디셔닝 조건이 drift 를 바꾸는가" 를 검정한다.

설계 결정 (사용자 확정)
    1. 주 모델은 선형. 단 곡률(2차항)과 First Wafer Effect 더미를 각각
       중첩 F 검정으로 진단만 하고, 유의할 때만 각주로 언급한다.
       lot 당 10장뿐이라 지수 점근 모델(모수 3개)은 시상수 CI 가 발산한다.
    2. 웨이퍼가 적은 lot(9점 lot7=5장, 89점 lot7=6장·lot10=4장)도 전부 포함.
       대신 표·그림에 n 을 명시하고 신뢰구간 폭으로 불확실성을 드러낸다.
       SiO2 조건은 lot 7·8·10 뿐이라 제외하면 재질 비교 자체가 불가능하다.
    3. 선택비 부호 불일치(9점 +, 89점 −)는 IDW 보간점 157개를 제외하고
       재적합해 원인을 확인한다.
    4. 조건 비교는 89점이 주(SiO2 수준 3개 lot 확보), 9점은 교차검증.

출력
    data/processed/drift_slopes_{9,89}pt.csv
    data/processed/drift_summary.csv
    data/processed/drift_interpolation_check.csv
    figures/drift_trend.png
    figures/drift_slope_by_condition.png
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

COND_COLOR = {"Chuck": "#4C72B0", "Si": "#55A868", "SiO2": "#DD8452"}
VALUE = "si_etch_mean"


def find(name):
    for d in DATA_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def style():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def load(n_points):
    return pd.read_csv(PROC / f"wafers_{n_points}pt.csv")


# ---------------------------------------------------------------- 기울기 추정


def lot_slopes(w, value=VALUE):
    """lot 별 단순 선형회귀. 웨이퍼 4장 lot 도 포함하되 n 을 남긴다."""
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
                "ci_width": hi - lo,
                "p_value": fit.pvalues["wafer_number"],
                "r2": fit.rsquared,
                "pct_per_10w": 100 * fit.params["wafer_number"] * 9 / first,
            }
        )
    return pd.DataFrame(rows).sort_values("lot_number").reset_index(drop=True)


def pooled_slope(w, value=VALUE):
    """lot 을 고정효과로 흡수한 공통 기울기 (lot 간 수준 차이 제거)."""
    fit = smf.ols(f"{value} ~ C(lot_number) + wafer_number", data=w).fit()
    lo, hi = fit.conf_int().loc["wafer_number"]
    return fit.params["wafer_number"], lo, hi, fit.pvalues["wafer_number"]


# ---------------------------------------------------------------- 형태 진단


def shape_diagnostics(w, value=VALUE):
    """선형이 충분한가 — 2차항과 First Wafer Effect 더미를 중첩 F 검정.

    lot 고정효과를 넣은 상태에서 항을 하나씩 추가하고 anova_lm 으로 비교한다.
    (lot 은 웨이퍼 순번과 교차 설계라 3단계의 rank deficient 문제가 없다)
    """
    d = w.copy()
    d["w2"] = d["wafer_number"] ** 2
    d["is_first"] = (d["wafer_number"] == 1).astype(int)

    m_lin = smf.ols(f"{value} ~ C(lot_number) + wafer_number", data=d).fit()
    m_quad = smf.ols(f"{value} ~ C(lot_number) + wafer_number + w2", data=d).fit()
    m_fwe = smf.ols(f"{value} ~ C(lot_number) + wafer_number + is_first", data=d).fit()

    out = {}
    for name, m, term in (("quadratic", m_quad, "w2"), ("first_wafer", m_fwe, "is_first")):
        tab = sm.stats.anova_lm(m_lin, m)
        out[name] = {
            "F": tab["F"].iloc[1],
            "p": tab["Pr(>F)"].iloc[1],
            "d_aic": m.aic - m_lin.aic,
            "extra_coef": m.params[term],
        }
    out["linear"] = {"aic": m_lin.aic, "r2": m_lin.rsquared}
    return out


# ---------------------------------------------------------------- 조건 비교


def condition_test(sl):
    """lot 별 기울기를 종속변수로 한 일원분산분석.

    표본이 lot 8~10개뿐이라 검정력이 낮다. 유의하지 않다는 결과를
    '차이 없음'으로 읽지 않도록 탐지 가능 최소 효과크기도 함께 반환한다.
    """
    out = {}
    for factor in ("cond_type", "cond_rep"):
        if sl[factor].nunique() < 2:
            continue
        fit = smf.ols(f"slope ~ C({factor})", data=sl).fit()
        grp = sl.groupby(factor)["slope"].agg(["mean", "std", "size"])
        n_per = sl.groupby(factor).size().mean()
        # 2군 비교, 유의수준 0.05 / 검정력 0.8 기준 근사 (2*1.4 = 2.8)
        mde = 2.8 * np.sqrt(fit.mse_resid) * np.sqrt(2 / n_per)
        out[factor] = {"F": fit.fvalue, "p": fit.f_pvalue, "groups": grp, "mde": mde}
    return out


def level_test(sl):
    """수준(lot 평균 식각량)에 대한 조건 효과. 날짜와 교락되어 있음에 주의."""
    out = {}
    for factor in ("cond_type", "cond_rep"):
        fit = smf.ols(f"level_mean ~ C({factor})", data=sl).fit()
        out[factor] = (fit.fvalue, fit.f_pvalue, sl.groupby(factor)["level_mean"].mean())
    return out


def repeatability(sl):
    """lot 1 과 lot 9 는 동일 조건(Chuck 3x), 50일 간격 — 재현성 벤치마크."""
    if not {1, 9}.issubset(set(sl["lot_number"])):
        return None
    a, b = (sl[sl.lot_number == k].iloc[0] for k in (1, 9))
    return {
        "level_1": a["level_mean"],
        "level_9": b["level_mean"],
        "level_diff": b["level_mean"] - a["level_mean"],
        "slope_1": a["slope"],
        "slope_9": b["slope"],
        "slope_diff": b["slope"] - a["slope"],
        "days": (pd.to_datetime(b["date"]) - pd.to_datetime(a["date"])).days,
    }


# ---------------------------------------------------- IDW 보간 영향 (89점 전용)


def _wafer_means(d):
    m = (
        d.groupby(["lot_number", "wafer_number"])[["si_etch", "oxide_etch"]]
        .mean()
        .rename(columns={"si_etch": "si_etch_mean", "oxide_etch": "oxide_etch_mean"})
        .reset_index()
    )
    m["selectivity"] = m["si_etch_mean"] / m["oxide_etch_mean"]
    return m


def selectivity_forensics():
    """선택비 추세 부호가 9점(+0.179)과 89점(−0.158)에서 반대인 원인 추적.

    용의자는 둘이다.
        (a) IDW 보간점 157개 — 89점의 postox_thickness 일부가 추정값
        (b) 계측 자체의 차이 — 9점과 89점은 장비가 다르고 좌표 집합도 다르다

    (b) 는 다시 '좌표 집합(grid)' 과 '장비' 로 쪼갤 수 있다. 9점 좌표가 89점
    좌표의 진부분집합이므로, 89점에서 그 9개 좌표만 뽑아 재적합하면 grid 효과가
    제거되고 순수한 장비 효과만 남는다. 세 조건을 나란히 비교한다.
    """
    d89 = pd.read_csv(find("Si_Oxide_etch_89_points.csv")).dropna(subset=["experiment_key"])
    d9 = pd.read_csv(find("Si_Oxide_etch_9_points.csv")).dropna(subset=["experiment_key"])
    d89["is_interp"] = d89["postox_thickness_nan"].isna()

    loc9 = set(zip(d9["X"], d9["Y"]))
    assert loc9 <= set(zip(d89["X"], d89["Y"])), "9점 좌표가 89점의 부분집합이 아니다"
    on_grid = [xy in loc9 for xy in zip(d89["X"], d89["Y"])]

    sets = {
        "89pt 전체": d89,
        "89pt 보간점제외": d89[~d89["is_interp"]],
        "89pt→9점grid": d89[on_grid],
        "9pt 원본": d9,
    }

    rows = []
    for tag, d in sets.items():
        m = _wafer_means(d)
        for v in ("si_etch_mean", "oxide_etch_mean", "selectivity"):
            s, lo, hi, p = pooled_slope(m, v)
            rows.append({"set": tag, "n_wafers": len(m), "value": v, "slope": s,
                         "ci_lo": lo, "ci_hi": hi, "p": p,
                         "wafer_mean_sd": m[v].std(), "level": m[v].mean()})
    n_by_lot = d89.groupby("lot_number")["is_interp"].sum()
    return pd.DataFrame(rows), int(d89["is_interp"].sum()), n_by_lot[n_by_lot > 0]


# ---------------------------------------------------------------- 보고


def report(n_points):
    w = load(n_points)
    sl = lot_slopes(w)
    slope, lo, hi, p = pooled_slope(w)
    base = w[VALUE].mean()

    print(f"\n{'=' * 74}")
    print(f"=== {n_points}점 : lot {w.lot_number.nunique()}개 / 웨이퍼 {len(w)}장 ===")
    print(f"{'=' * 74}")

    print("\n[1] lot 별 drift 기울기")
    cols = ["lot_number", "cond_label", "n_wafers", "level_mean", "slope",
            "ci_lo", "ci_hi", "p_value", "r2", "pct_per_10w"]
    print(sl[cols].round(4).to_string(index=False))

    print("\n[2] 공통 기울기 (lot 고정효과 흡수)")
    print(f"    {slope:+.4f} µm/웨이퍼  [95% CI {lo:+.4f}, {hi:+.4f}]  p={p:.2e}")
    print(f"    10장 누적 {slope * 9:+.3f} µm ({100 * slope * 9 / base:+.2f}% of {base:.2f} µm)")
    print(f"    개별 lot 중 p<0.05 : {(sl.p_value < 0.05).sum()}/{len(sl)}"
          f",  기울기 음수 : {(sl.slope < 0).sum()}/{len(sl)}")

    print("\n[3] 형태 진단 — 선형으로 충분한가")
    diag = shape_diagnostics(w)
    for k in ("quadratic", "first_wafer"):
        d = diag[k]
        verdict = "유의 (선형 부족)" if d["p"] < 0.05 else "기각 못 함 (선형으로 충분)"
        print(f"    +{k:12s} F={d['F']:6.2f}  p={d['p']:.3f}  ΔAIC={d['d_aic']:+.1f}"
              f"  계수={d['extra_coef']:+.4f}  → {verdict}")

    print("\n[4] 컨디셔닝 조건별 기울기 비교")
    for factor, r in condition_test(sl).items():
        print(f"    {factor}: F={r['F']:.2f}  p={r['p']:.3f}"
              f"   (검정력 80% 탐지한계 ±{r['mde']:.3f} µm/장)")
        for line in r["groups"].round(4).to_string().split("\n"):
            print("      " + line)

    print("\n[5] 수준(lot 평균 식각량)에 대한 조건 효과  ※ 날짜와 교락")
    for factor, (f, pv, means) in level_test(sl).items():
        print(f"    {factor}: F={f:.2f}  p={pv:.3f}   {means.round(3).to_dict()}")

    rep = repeatability(sl)
    if rep:
        print(f"\n[6] 재현성 (lot 1 vs lot 9, 동일 조건 Chuck_3x, {rep['days']}일 간격)")
        print(f"    수준   {rep['level_1']:.3f} → {rep['level_9']:.3f} µm   (차이 {rep['level_diff']:+.3f})")
        print(f"    기울기 {rep['slope_1']:+.4f} → {rep['slope_9']:+.4f} µm/장 (차이 {rep['slope_diff']:+.4f})")

    ssl = lot_slopes(w, "selectivity")
    sl["value"] = VALUE
    ssl["value"] = "selectivity"
    pd.concat([sl, ssl]).to_csv(PROC / f"drift_slopes_{n_points}pt.csv", index=False)

    summary = {
        "n_points": n_points,
        "n_lots": w.lot_number.nunique(),
        "n_wafers": len(w),
        "pooled_slope": slope,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": p,
        "cum_10w_um": slope * 9,
        "cum_10w_pct": 100 * slope * 9 / base,
        "quad_p": diag["quadratic"]["p"],
        "fwe_p": diag["first_wafer"]["p"],
    }
    return w, sl, summary


# ---------------------------------------------------------------- 그림


def plot_trend(data):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (n, (w, sl, _)) in zip(axes, data.items()):
        for lot, g in w.groupby("lot_number"):
            g = g.sort_values("wafer_number")
            dev = g[VALUE] - g[VALUE].mean()
            c = COND_COLOR[g["cond_type"].iloc[0]]
            ax.plot(g["wafer_number"], dev, "o-", color=c, alpha=0.45, ms=4, lw=1)
        pooled = w.assign(dev=w[VALUE] - w.groupby("lot_number")[VALUE].transform("mean"))
        fit = smf.ols("dev ~ wafer_number", data=pooled).fit()
        x = np.arange(1, int(w["wafer_number"].max()) + 1)
        pred = fit.get_prediction(pd.DataFrame({"wafer_number": x})).summary_frame(alpha=0.05)
        ax.plot(x, pred["mean"], "k-", lw=2.5,
                label=f"공통 추세 {fit.params['wafer_number']:+.3f} µm/장")
        ax.fill_between(x, pred["mean_ci_lower"], pred["mean_ci_upper"], color="k", alpha=0.12)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel("lot 내 웨이퍼 순번")
        ax.set_xticks(x)
        ax.set_title(f"{n}점 계측  (lot {w.lot_number.nunique()}개 / {len(w)}장)")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].set_ylabel("Si 식각 깊이 - lot 평균 (µm)")
    handles = [plt.Line2D([], [], color=c, marker="o", ls="-", label=f"컨디셔닝 {k}")
               for k, c in COND_COLOR.items()]
    axes[1].legend(handles=handles + axes[1].get_legend_handles_labels()[0],
                   frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("lot 내 웨이퍼 순번에 따른 식각 깊이 감소 (drift)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "drift_trend.png", dpi=150)


def _forest(ax, sl, pooled, title):
    sl = sl.sort_values("lot_number").reset_index(drop=True)
    y = np.arange(len(sl))[::-1]
    for yi, (_, r) in zip(y, sl.iterrows()):
        c = COND_COLOR[r["cond_type"]]
        ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, lw=2, alpha=0.85)
        ax.plot(r["slope"], yi, "o", color=c, ms=7, zorder=3)
    s, lo, hi = pooled
    ax.axvspan(lo, hi, color="k", alpha=0.10, zorder=0)
    ax.axvline(s, color="k", lw=1.6, label=f"공통 {s:+.3f} µm/장")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"lot {int(l)}  {c}  n={int(n)}"
                        for l, c, n in zip(sl["lot_number"], sl["cond_label"], sl["n_wafers"])],
                       fontsize=8)
    ax.set_xlabel("drift 기울기 (µm/웨이퍼)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(-0.6, len(sl) + 0.2)  # 범례 자리 확보


def plot_slopes(data):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (n, (w, sl, summ)) in zip(axes[:2], data.items()):
        _forest(ax, sl, (summ["pooled_slope"], summ["ci_lo"], summ["ci_hi"]),
                f"{n}점 계측 — lot 별 기울기와 95% CI")

    ax = axes[2]
    sl89 = data[89][1]
    xs, labels = [], []
    for i, (t, g) in enumerate(sl89.groupby("cond_type")):
        jitter = np.linspace(-0.12, 0.12, len(g))
        ax.scatter(i + jitter, g["slope"], color=COND_COLOR[t], s=55, zorder=3,
                   edgecolor="white", linewidth=0.8)
        m = g["slope"].mean()
        ax.plot([i - 0.25, i + 0.25], [m, m], color=COND_COLOR[t], lw=2.5)
        labels.append(f"{t}\n(lot {len(g)}개)")
        xs.append(i)
    pooled89 = data[89][2]["pooled_slope"]
    ax.axhline(pooled89, color="k", lw=1.2, label=f"전체 공통 {pooled89:+.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("drift 기울기 (µm/웨이퍼)")
    ax.set_title("89점 — 컨디셔닝 재질별 기울기\n(군간 차이 유의하지 않음)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("컨디셔닝 조건은 drift 기울기를 바꾸지 못한다", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "drift_slope_by_condition.png", dpi=150)


# ---------------------------------------------------------------- main


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    data = {n: report(n) for n in (9, 89)}

    print(f"\n{'=' * 74}")
    print("=== [7] 선택비 부호 불일치 원인 추적 ===")
    print(f"{'=' * 74}")
    tab, n_interp, by_lot = selectivity_forensics()
    print(f"IDW 보간점 {n_interp}개 / 7832점, lot 별 분포: {by_lot.to_dict()}")
    for v in ("si_etch_mean", "oxide_etch_mean", "selectivity"):
        print(f"\n  [{v}]")
        sub = tab[tab["value"] == v]
        print(sub[["set", "n_wafers", "slope", "ci_lo", "ci_hi", "p", "level", "wafer_mean_sd"]]
              .round(5).to_string(index=False))

    ox = tab[tab["value"] == "oxide_etch_mean"].set_index("set")
    print(f"\n  → 산화막 측정 노이즈비 (89pt→9점grid / 9pt) = "
          f"{ox.loc['89pt→9점grid', 'wafer_mean_sd'] / ox.loc['9pt 원본', 'wafer_mean_sd']:.1f}배")

    pd.DataFrame([d[2] for d in data.values()]).to_csv(PROC / "drift_summary.csv", index=False)
    tab.to_csv(PROC / "drift_selectivity_forensics.csv", index=False)

    plot_trend(data)
    plot_slopes(data)
    print("\n-> figures/drift_trend.png, figures/drift_slope_by_condition.png")
    print("-> data/processed/drift_slopes_{9,89}pt.csv, drift_summary.csv,")
    print("   drift_selectivity_forensics.csv")


if __name__ == "__main__":
    main()
