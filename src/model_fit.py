"""4단계: si_etch 가상계측(VM) 모델 학습·검증 (증거 생성기).

계약: 이 스크립트는 **표/수치만** 만든다. "핵심 원인은 X / 이 피처를 써라"
같은 결론 문장은 넣지 않는다 — 판단은 사용자.

입력 : data/processed/model_table.parquet  (75웨이퍼 × 108피처 + y + lot + 컨디셔닝)
타깃 : si_etch_mean  (9점 중심 대표값, µm)
그룹 : lot_number    → GroupKFold. 8 lot이므로 n_splits=8 = Leave-One-Lot-Out(LOLO).
       (CLAUDE.md: 랜덤 분할 금지. 형제 웨이퍼 누수로 성능 거짓 부풀림.)

전처리(features 파이프 노트 반영):
  - EpdIntensity 연속피처(매 lot 첫 장만 ~18.5, 나머지 상수) → is_first_wafer 이진 플래그.
  - 저분산 near-constant 피처 → 분산필터 제거(무엇이 빠졌는지 보고).
    (표준화 시 std≈0 나눗셈이 단일 스파이크를 폭발시키는 아티팩트 차단.)

핵심 진단:
  - in-lot 학습 R²(전체 적합, 낙관 상한) vs cross-lot LOLO-CV R²(정직한 일반화).
    둘의 격차 = between-lot 전이 여부의 증거.
  - 베이스라인: 전역 평균 / train-lot 평균(=lot 레벨만 맞추기) 대비 CV 성능 위치.

산출:
  data/processed/model_cv_results.csv   (모델별 CV 지표)
  data/processed/model_coefficients.csv (선형 표준화 계수 랭킹)
  figures/fig_model_cv.png              (train vs CV 격차 / LOLO 예측-실측)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    MODEL_TABLE_PARQUET,
    ensure_output_dirs,
)
from model import feature_columns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TARGET = "si_etch_mean"
GROUP_COL = "lot_number"
VAR_THRESHOLD = 0.01     # MinMax[0,1] 스케일 후 분산 하한 (near-constant 제거)


# ── 전처리 ────────────────────────────────────────────────────────
def build_design_matrix(df: pd.DataFrame):
    """(X, y, groups, kept_feats, drop_report) 반환.

    EpdIntensity 연속피처 → is_first_wafer 플래그, 저분산 피처는 분산필터 제거.
    """
    feats = feature_columns(df)
    epd = [f for f in feats if f.startswith("EpdIntensity")]
    cont = [f for f in feats if f not in epd]

    # 분산필터: MinMax[0,1] 후 분산 < VAR_THRESHOLD 인 near-constant 제거.
    #  (MinMax 는 스케일/이동 불변 → slope(평균≈0) 피처도 공정하게 판정)
    Xc = df[cont].to_numpy(float)
    mm = MinMaxScaler().fit_transform(Xc)
    vt = VarianceThreshold(VAR_THRESHOLD).fit(mm)
    keep_mask = vt.get_support()
    kept = [c for c, k in zip(cont, keep_mask) if k]
    dropped_lowvar = [c for c, k in zip(cont, keep_mask) if not k]

    X = df[kept].copy()
    X["is_first_wafer"] = (df["wafer_number"] == 1).astype(int)

    y = df[TARGET].to_numpy(float)
    groups = df[GROUP_COL].to_numpy()

    drop_report = {
        "epd_to_flag": epd,
        "lowvar_dropped": dropped_lowvar,
        "n_kept_feats": len(kept),
    }
    return X, y, groups, kept + ["is_first_wafer"], drop_report


# ── 모델 정의 (규제선형 + RF, 딥러닝 금지) ────────────────────────
def make_models():
    """{name: (pipeline, param_grid)}  — 파이프라인에 스케일러 포함(폴드 내 적합)."""
    alphas = np.logspace(-3, 2, 12)
    l1 = [0.2, 0.5, 0.8]
    lin = lambda est: Pipeline([("sc", StandardScaler()), ("est", est)])
    return {
        "Ridge": (lin(Ridge(max_iter=50000)),
                  {"est__alpha": alphas}),
        "Lasso": (lin(Lasso(max_iter=100000)),
                  {"est__alpha": np.logspace(-4, 0, 12)}),
        "ElasticNet": (lin(ElasticNet(max_iter=100000)),
                       {"est__alpha": np.logspace(-4, 0, 10),
                        "est__l1_ratio": l1}),
        "RandomForest": (Pipeline([("est", RandomForestRegressor(
                            n_estimators=200, random_state=0, n_jobs=1))]),
                         {"est__max_depth": [2, 3, None],
                          "est__min_samples_leaf": [1, 2]}),
    }


# ── LOLO(Leave-One-Lot-Out) 중첩 CV ───────────────────────────────
def lolo_predict(pipe, grid, X, y, groups):
    """바깥 LOLO, 안쪽 GroupKFold 로 하이퍼파라미터 튜닝 → OOF 예측 벡터.

    누수 방지: 스케일링·alpha 선택 모두 held-out lot 을 보지 않고 학습쪽에서만.
    """
    oof = np.full(len(y), np.nan)
    base_lot_mean = np.full(len(y), np.nan)   # train-lot 평균 베이스라인
    uniq = np.array(sorted(np.unique(groups)))
    for test_lot in uniq:
        te = groups == test_lot
        tr = ~te
        n_train_lots = len(np.unique(groups[tr]))
        inner = GroupKFold(n_splits=min(n_train_lots, 5))
        gs = GridSearchCV(pipe, grid, cv=inner,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X.iloc[tr], y[tr], groups=groups[tr])
        oof[te] = gs.predict(X.iloc[te])
        base_lot_mean[te] = y[tr].mean()
    return oof, base_lot_mean


def fit_all(df: pd.DataFrame):
    X, y, groups, cols, rep = build_design_matrix(df)
    models = make_models()

    rows = []
    oof_store = {}
    # 베이스라인들
    dummy_pred = np.full(len(y), y.mean())          # 전역 평균(참고: CV에선 아래 lot평균)
    rows.append(("Baseline_globalmean", np.nan,
                 r2_score(y, dummy_pred), root_mean_squared_error(y, dummy_pred),
                 np.nan))

    lot_mean_pred = None
    for name, (pipe, grid) in models.items():
        # in-lot 학습(전체 적합, 낙관 상한): 같은 데이터 예측
        gs_full = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=8),
                               scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs_full.fit(X, y, groups=groups)
        train_pred = gs_full.predict(X)
        train_r2 = r2_score(y, train_pred)

        # cross-lot LOLO-CV
        oof, base_lot_mean = lolo_predict(pipe, grid, X, y, groups)
        cv_r2 = r2_score(y, oof)
        cv_rmse = root_mean_squared_error(y, oof)
        oof_store[name] = oof
        if lot_mean_pred is None:
            lot_mean_pred = base_lot_mean

        rows.append((name, train_r2, cv_r2, cv_rmse, train_r2 - cv_r2))

    # train-lot 평균 베이스라인 (LOLO 문맥의 진짜 하한)
    rows.insert(1, ("Baseline_trainlotmean", np.nan,
                    r2_score(y, lot_mean_pred),
                    root_mean_squared_error(y, lot_mean_pred), np.nan))

    res = pd.DataFrame(rows, columns=[
        "model", "train_R2(in-lot)", "CV_R2(LOLO)", "CV_RMSE_um", "R2_gap"])
    return res, X, y, groups, cols, rep, oof_store


# ── 선형 계수 랭킹 (표준화 계수, 절댓값순) ─────────────────────────
def linear_coefficients(df, X, y, groups, cols):
    """Ridge/Lasso 를 전체 적합 후 표준화 계수 랭킹 (선지정 없이 등장 순서만)."""
    out = {}
    for name, est in [("Ridge", Ridge(max_iter=50000)),
                      ("Lasso", Lasso(max_iter=100000))]:
        pipe = Pipeline([("sc", StandardScaler()), ("est", est)])
        gs = GridSearchCV(
            pipe,
            {"est__alpha": (np.logspace(-3, 2, 12) if name == "Ridge"
                            else np.logspace(-4, 0, 12))},
            cv=GroupKFold(n_splits=8),
            scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X, y, groups=groups)
        coef = gs.best_estimator_.named_steps["est"].coef_
        out[name] = pd.Series(coef, index=cols)
        out[f"{name}_alpha"] = gs.best_params_["est__alpha"]
    tbl = pd.DataFrame({"feature": cols,
                        "Ridge_coef": out["Ridge"].values,
                        "Lasso_coef": out["Lasso"].values})
    tbl["abs_max"] = tbl[["Ridge_coef", "Lasso_coef"]].abs().max(axis=1)
    tbl = tbl.sort_values("abs_max", ascending=False).reset_index(drop=True)
    return tbl, out["Ridge_alpha"], out["Lasso_alpha"]


# ── 그림 ──────────────────────────────────────────────────────────
def make_figure(res, X, y, groups, oof_store):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))

    # 좌: train R² vs CV R² 격차 (베이스라인 제외)
    m = res[~res["model"].str.startswith("Baseline")].copy()
    xpos = np.arange(len(m))
    axL.bar(xpos - 0.2, m["train_R2(in-lot)"], width=0.4,
            color="tab:gray", label="train R2 (in-lot, 낙관상한)")
    axL.bar(xpos + 0.2, m["CV_R2(LOLO)"], width=0.4,
            color="crimson", label="CV R2 (cross-lot LOLO)")
    axL.axhline(0, color="k", lw=0.8)
    axL.set_xticks(xpos); axL.set_xticklabels(m["model"], rotation=15, fontsize=8)
    axL.set_ylabel("R2")
    axL.set_title("in-lot 적합 vs cross-lot CV 격차")
    axL.grid(axis="y", alpha=0.3); axL.legend(fontsize=8)

    # 우: 최고 CV 모델의 LOLO 예측-실측 (lot 색)
    best = m.loc[m["CV_R2(LOLO)"].idxmax(), "model"]
    oof = oof_store[best]
    uniq = sorted(np.unique(groups))
    cmap = plt.get_cmap("tab10")
    for i, lot in enumerate(uniq):
        s = groups == lot
        axR.scatter(y[s], oof[s], s=40, color=cmap(i % 10), label=f"lot{lot}")
    lo, hi = min(y.min(), np.nanmin(oof)), max(y.max(), np.nanmax(oof))
    axR.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    axR.set_xlabel("실측 si_etch_mean (µm)")
    axR.set_ylabel("LOLO 예측 (µm)")
    axR.set_title(f"최고 CV 모델={best}: cross-lot 예측 vs 실측")
    axR.grid(alpha=0.3); axR.legend(fontsize=7, ncol=2)

    fig.suptitle("Fig. VM 모델 4단계: GroupKFold(lot) Leave-One-Lot-Out 검증")
    fig.tight_layout()
    p = FIGURES_DIR / "fig_model_cv.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p, best


def main():
    ap = argparse.ArgumentParser(description="VM 모델 학습·검증 (4단계)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    ensure_output_dirs()
    df = pd.read_parquet(MODEL_TABLE_PARQUET)
    print(f"입력: {len(df)}웨이퍼 × {len(feature_columns(df))}피처, "
          f"lot {sorted(df[GROUP_COL].unique())}")

    res, X, y, groups, cols, rep, oof_store = fit_all(df)

    print("\n=== 전처리 ===")
    print(f"  EpdIntensity 연속피처 → is_first_wafer 플래그: {rep['epd_to_flag']}")
    print(f"  분산필터 제거({len(rep['lowvar_dropped'])}): {rep['lowvar_dropped']}")
    print(f"  최종 피처 수: {rep['n_kept_feats']} + is_first_wafer = {len(cols)}")

    print("\n=== 모델별 성능 (계약: 수치만, 해석은 사용자) ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.3f}"
                        if pd.notna(x) else "  --"))

    coef_tbl, ra, la = linear_coefficients(df, X, y, groups, cols)
    print(f"\n=== 선형 표준화 계수 랭킹 상위 15 "
          f"(Ridge α={ra:.4g}, Lasso α={la:.4g}) ===")
    print(coef_tbl.head(15).to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))
    cap = coef_tbl[coef_tbl["feature"].str.contains("Capacitor")]
    if len(cap):
        print(f"  [참고] Capacitor 계수 위치(순위):")
        for _, r in cap.iterrows():
            rank = coef_tbl.index[coef_tbl["feature"] == r["feature"]][0] + 1
            print(f"    {rank:3d}/{len(coef_tbl)}  {r['feature']:40s} "
                  f"Ridge={r['Ridge_coef']:+.4f} Lasso={r['Lasso_coef']:+.4f}")

    p, best = make_figure(res, X, y, groups, oof_store)
    print(f"\nsaved {p}  (최고 CV 모델={best})")

    if not args.no_save:
        res.to_csv(DATA_PROCESSED / "model_cv_results.csv", index=False)
        coef_tbl.to_csv(DATA_PROCESSED / "model_coefficients.csv", index=False)
        print(f"saved {DATA_PROCESSED / 'model_cv_results.csv'}")
        print(f"saved {DATA_PROCESSED / 'model_coefficients.csv'}")


if __name__ == "__main__":
    main()
