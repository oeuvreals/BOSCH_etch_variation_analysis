"""4단계 진단: LOLO CV R²(=0.815)를 between-lot(레벨차) vs within-lot(드리프트)로 분해.

계약: 수치/그림만. "모델이 드리프트를 잡았다/못잡았다" 결론은 사용자.

배경: si_etch 산포 = lot간 레벨차(컨디셔닝/노화) + lot내 런순번 드리프트(-1.7%).
전체 LOLO R²는 이 둘을 섞는다. 프로젝트 실제 타깃은 후자(드리프트).
  - between R² : 모델이 lot-to-lot 평균 레벨을 재현하나? (컨디셔닝 교락 주의)
  - within  R²: lot 평균 제거 후 런순번 변동(드리프트)을 추종하나?
  - 추가: lot별 실측 드리프트 slope vs 예측 drift slope (방향/크기 일치?)

Ridge(가벼움)만 LOLO 재현 → oof. 산출:
  data/processed/model_within_between.csv, figures/fig_model_within_between.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, FIGURES_DIR, MODEL_TABLE_PARQUET, ensure_output_dirs
from model_fit import build_design_matrix

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def ridge_oof(X, y, groups):
    oof = np.full(len(y), np.nan)
    for test_lot in sorted(np.unique(groups)):
        te = groups == test_lot
        tr = ~te
        inner = GroupKFold(n_splits=min(len(np.unique(groups[tr])), 5))
        gs = GridSearchCV(
            Pipeline([("sc", StandardScaler()), ("est", Ridge(max_iter=50000))]),
            {"est__alpha": np.logspace(-3, 2, 12)}, cv=inner,
            scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X.iloc[tr], y[tr], groups=groups[tr])
        oof[te] = gs.predict(X.iloc[te])
    return oof


def wslope(order, vals):
    t = np.asarray(order, float); t = t - t.mean()
    d = (t * t).sum()
    return float((t * (np.asarray(vals, float) - np.mean(vals))).sum() / d) if d else np.nan


def r2_within_masked(y, oof, groups, mask):
    """mask 부분집합에서 lot 평균 제거 후(=드리프트 성분) within R²."""
    g = groups[mask]; yy = y[mask]; pp = oof[mask]
    ym = {l: yy[g == l].mean() for l in np.unique(g)}
    pm = {l: pp[g == l].mean() for l in np.unique(g)}
    y_w = np.array([yy[i] - ym[g[i]] for i in range(len(yy))])
    p_w = np.array([pp[i] - pm[g[i]] for i in range(len(yy))])
    return 1 - np.sum((y_w - p_w) ** 2) / np.sum(y_w ** 2)


def main():
    ensure_output_dirs()
    df = pd.read_parquet(MODEL_TABLE_PARQUET)
    X, y, groups, cols, _ = build_design_matrix(df)
    order = df["wafer_number"].to_numpy(float)

    oof = ridge_oof(X, y, groups)
    overall = r2_score(y, oof)

    lots = np.array(sorted(np.unique(groups)))
    ymean = {l: y[groups == l].mean() for l in lots}
    pmean = {l: oof[groups == l].mean() for l in lots}
    grand = y.mean()

    # between-lot R² (lot 평균 레벨 재현)
    ss_tot_b = sum((groups == l).sum() * (ymean[l] - grand) ** 2 for l in lots)
    ss_res_b = sum((groups == l).sum() * (ymean[l] - pmean[l]) ** 2 for l in lots)
    r2_between = 1 - ss_res_b / ss_tot_b

    # within-lot R² (lot 평균 제거 후 = 드리프트 성분)
    y_w = np.array([y[i] - ymean[groups[i]] for i in range(len(y))])
    p_w = np.array([oof[i] - pmean[groups[i]] for i in range(len(y))])
    r2_within = 1 - np.sum((y_w - p_w) ** 2) / np.sum(y_w ** 2)

    # lot별 드리프트 slope: 실측 vs 예측 (+ 순번 범위 = S커브상 위치)
    rows = []
    for l in lots:
        s = groups == l
        rows.append((int(l), int(s.sum()),
                     int(order[s].min()), int(order[s].max()),
                     ymean[l], pmean[l],
                     wslope(order[s], y[s]), wslope(order[s], oof[s])))
    dd = pd.DataFrame(rows, columns=[
        "lot", "n", "wafer_min", "wafer_max", "y_mean", "pred_mean",
        "y_driftslope", "pred_driftslope"])
    slope_r = np.corrcoef(dd["y_driftslope"], dd["pred_driftslope"])[0, 1]

    # lot7 민감도: n=5 반쪽 + 순번1~5(S커브 평평 초반)만 → 별도 병기(삭제X)
    def _metrics(mask):
        return (r2_score(y[mask], oof[mask]),
                r2_within_masked(y, oof, groups, mask),
                float(np.sqrt(np.mean((y[mask] - oof[mask]) ** 2))))
    allm = np.ones(len(y), bool)
    no7 = groups != 7
    sens = pd.DataFrame(
        [("전체_8lot", int(allm.sum()), *_metrics(allm)),
         ("lot7제외_7lot", int(no7.sum()), *_metrics(no7))],
        columns=["subset", "n_wafers", "LOLO_R2", "within_R2", "RMSE_um"])

    print(f"전체 LOLO R²(Ridge, 재현) = {overall:+.3f}")
    print(f"  between-lot R²(레벨 재현)  = {r2_between:+.3f}")
    print(f"  within-lot  R²(드리프트)   = {r2_within:+.3f}")
    print(f"  lot별 드리프트 slope 상관(실측 vs 예측) r = {slope_r:+.3f}"
          f"  (실측 slope 대역 [{dd.y_driftslope.min():+.3f},"
          f" {dd.y_driftslope.max():+.3f}] 좁음 → 상관 spread 부족)")
    print("\nlot별 (계약: 수치만):")
    print(dd.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== lot7 민감도 (n=5, 순번1~5=S커브 초반만; 삭제X 병기) ===")
    print(sens.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    dd.to_csv(DATA_PROCESSED / "model_within_between.csv", index=False)
    sens.to_csv(DATA_PROCESSED / "model_lot7_sensitivity.csv", index=False)

    # ── 그림 2패널 ──
    fig, (axB, axW) = plt.subplots(1, 2, figsize=(12, 5.2))
    cmap = plt.get_cmap("tab10")
    for i, l in enumerate(lots):
        axB.scatter(ymean[l], pmean[l], s=70, color=cmap(i % 10), label=f"lot{l}")
    lo = min(min(ymean.values()), min(pmean.values()))
    hi = max(max(ymean.values()), max(pmean.values()))
    axB.plot([lo, hi], [lo, hi], "--", color="gray")
    axB.set_xlabel("실측 lot 평균 (µm)"); axB.set_ylabel("예측 lot 평균 (µm)")
    axB.set_title(f"between-lot 레벨 재현  R2={r2_between:+.3f}")
    axB.grid(alpha=0.3); axB.legend(fontsize=7, ncol=2)

    for i, l in enumerate(lots):
        s = groups == l
        axW.scatter(y_w[s], p_w[s], s=35, color=cmap(i % 10), label=f"lot{l}")
    lo2 = min(y_w.min(), p_w.min()); hi2 = max(y_w.max(), p_w.max())
    axW.plot([lo2, hi2], [lo2, hi2], "--", color="gray")
    axW.axhline(0, color="k", lw=0.5); axW.axvline(0, color="k", lw=0.5)
    axW.set_xlabel("실측 lot내 편차 (µm, 드리프트)")
    axW.set_ylabel("예측 lot내 편차 (µm)")
    axW.set_title(f"within-lot 드리프트 추종  R2={r2_within:+.3f}")
    axW.grid(alpha=0.3); axW.legend(fontsize=7, ncol=2)

    fig.suptitle("Fig. LOLO R2 분해: 레벨차(between) vs 드리프트(within)")
    fig.tight_layout()
    p = FIGURES_DIR / "fig_model_within_between.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nsaved {p}")
    print(f"saved {DATA_PROCESSED / 'model_within_between.csv'}")
    print(f"saved {DATA_PROCESSED / 'model_lot7_sensitivity.csv'}")


if __name__ == "__main__":
    main()
