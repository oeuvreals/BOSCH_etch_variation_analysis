"""공정 요약 피처 → si_etch 가상계측(VM) 모델.

3단계(현재 책임): 모델 입력 테이블 조립.
  process_features.parquet(X, 108피처) 와 wafers_9pt.csv(y + lot + 컨디셔닝) 를
  (date, wafer_number) 로 조인한다.
    - X      = 108 요약 피처 (채널 × mean/std/slope/range)
    - y      = si_etch_mean  (9점 중심 대표값)   ※ CLAUDE.md: 9점만, 89점 혼용 금지
    - groups = lot_number    ← GroupKFold 필수 (랜덤 분할 금지)
    - cond_* = 컨디셔닝(Chuck/Si/SiO2 × 1/3/9) : 교락 분석·계층화용

산출물: data/processed/model_table.parquet  (gitignore 대상 parquet)

4단계(예정): 이 테이블 위에 규제선형(Ridge/Lasso/ElasticNet)·RF 를 GroupKFold(lot)
으로 학습·검증. 딥러닝 금지(p≫n, 75웨이퍼). EpdIntensity 는 연속피처가 아니라
is_first_wafer 플래그로 처리, 저분산 range 피처는 분산필터로 제거(→ features 파이프 노트).

규칙: 조인은 (date, wafer_number) one-to-one 검증. 매칭 실패 시 조용히 빈 조인이
나므로 반드시 행 수를 확인한다.
"""
from __future__ import annotations

import argparse

import pandas as pd

from config import (
    MODEL_TABLE_PARQUET,
    PROCESS_FEATURES_PARQUET,
    WAFERS_9PT_CSV,
    ensure_output_dirs,
)

# 조인 키 / 타깃 / 그룹
MERGE_KEYS = ["date", "wafer_number"]
TARGET = "si_etch_mean"          # y (9점 중심 대표값)
GROUP_COL = "lot_number"         # GroupKFold groups

# 피처 컬럼 판별용 접미사 (process_features.py 와 일치).
FEATURE_SUFFIXES = ("__mean", "__std", "__slope", "__range")

# 측정 테이블에서 모델 테이블로 가져올 메타(피처 아님).
_MEAS_META = [
    "lot_number", "cond_type", "cond_rep", "cond_label",
    "si_etch_mean", "si_etch_std", "selectivity",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """108개 요약 피처 컬럼명(접미사 기준)."""
    return [c for c in df.columns if c.endswith(FEATURE_SUFFIXES)]


def build_model_table() -> pd.DataFrame:
    """X(공정 피처) ⋈ y(9점 측정) 조인 → 모델 입력 테이블.

    (date, wafer_number) one-to-one inner join. 9점이 없는 웨이퍼
    (lot7 6번째~, lot8, lot10)는 타깃이 없어 자동 제외된다.
    """
    feat = pd.read_parquet(PROCESS_FEATURES_PARQUET)
    meas = pd.read_csv(WAFERS_9PT_CSV)

    keep = MERGE_KEYS + [c for c in _MEAS_META if c in meas.columns]
    merged = feat.merge(
        meas[keep], on=MERGE_KEYS, how="inner", validate="one_to_one",
    )

    # 컬럼 정렬: 식별/메타 → 피처. (모델 코드가 앞부분만 봐도 되게)
    feats = feature_columns(merged)
    meta = [
        "lot_number", "wafer_number", "date", "group_name", "n_frames",
        "cond_type", "cond_rep", "cond_label",
        "si_etch_mean", "si_etch_std", "selectivity",
    ]
    meta = [c for c in meta if c in merged.columns]
    return merged[meta + sorted(feats)]


def main() -> None:
    ap = argparse.ArgumentParser(description="모델 입력 테이블 조립 (3단계)")
    ap.add_argument("--no-save", action="store_true",
                    help="parquet 저장 없이 요약만 출력")
    args = ap.parse_args()

    ensure_output_dirs()
    feat = pd.read_parquet(PROCESS_FEATURES_PARQUET)
    df = build_model_table()

    n_feat = len(feature_columns(df))
    dropped = len(feat) - len(df)
    print(f"조인 완료: {len(df)}웨이퍼 × {n_feat}피처 "
          f"(공정 {len(feat)}장 중 9점 없어 제외 {dropped}장)")
    print(f"타깃 {TARGET} 결측: {int(df[TARGET].isna().sum())}, "
          f"그룹 {GROUP_COL} 결측: {int(df[GROUP_COL].isna().sum())}")
    print(f"GroupKFold 그룹(lot) 수: {df[GROUP_COL].nunique()}  "
          f"→ {sorted(df[GROUP_COL].unique())}")
    print("\nlot별 웨이퍼 수:")
    print(df.groupby(GROUP_COL)["wafer_number"].count().to_string())
    print("\n컨디셔닝별 웨이퍼 수(교락 구조 점검):")
    print(df.groupby("cond_label")["wafer_number"].count().to_string())

    if not args.no_save:
        df.to_parquet(MODEL_TABLE_PARQUET, index=False)
        print(f"\n저장: {MODEL_TABLE_PARQUET}")


if __name__ == "__main__":
    main()
