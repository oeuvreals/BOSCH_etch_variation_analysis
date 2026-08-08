"""웨이퍼당 공정 시계열 → 요약 피처 테이블.

process_parse.read_wafer 로 웨이퍼를 1장씩(메모리 안전) 읽어, 27개 유효 채널
각각에서 요약 통계를 뽑아 웨이퍼당 1행으로 축약한다.
산출물: data/processed/process_features.parquet  (웨이퍼 행 × 피처 열)

피처 설계 (채널별 4종, 공정 의미 기준):
- mean  : 공정 중 평균 수준. 설정값/절대 드리프트.
- std   : 변동성. BOSCH 펄싱 채널은 스위칭 진폭이 여기에 반영된다.
- slope : 웨이퍼 '내부' 시간 기울기(단위 /s). 런 내부 드리프트 신호 → 챔버 drift 후보.
- range : robust 진폭 p95-p05. 펄싱 채널의 스위칭 폭 프록시(사이클 검출 없이 안정적).

이 모듈은 group_name/date/wafer_number + 피처까지만 책임진다.
date→lot 조인과 타깃(si_etch) 결합은 이후 단계(build/model)에서 처리한다.

규칙: 새 스크립트는 먼저 웨이퍼 1장으로 검증 → `python process_features.py --limit 1`.
"""
from __future__ import annotations

import argparse
import time as _time

import numpy as np
import pandas as pd

from config import PROCESS_FEATURES_PARQUET, ensure_output_dirs
from process_parse import (
    WaferSeries,
    list_wafer_groups,
    read_wafer,
    valid_features,
)

# 채널 이름 접두사. 컬럼 가독성을 위해 제거 (Stat3_Etch_MV_Gas1Flow → Gas1Flow).
_CHAN_PREFIX = "Stat3_Etch_MV_"

# 채널별로 뽑는 요약 통계 이름 (컬럼 접미사).
STAT_NAMES = ("mean", "std", "slope", "range")


def short_channel(name: str) -> str:
    """채널 컬럼명 축약: 공통 접두사 제거."""
    return name[len(_CHAN_PREFIX):] if name.startswith(_CHAN_PREFIX) else name


def _time_slope(times: np.ndarray, data: np.ndarray) -> np.ndarray:
    """채널별 최소제곱 시간 기울기 (단위 값/s).

    slope_j = cov(t, x_j) / var(t).  중심화 후 벡터화로 채널 동시 계산.
    times 는 상대 초. 분산이 0(정지 시간축)이면 0 반환.
    """
    t = times.astype("float64") - times.mean()
    denom = float((t * t).sum())
    if denom == 0.0:
        return np.zeros(data.shape[1], dtype="float64")
    xc = data.astype("float64") - data.mean(axis=0, keepdims=True)
    return (t[:, None] * xc).sum(axis=0) / denom


def wafer_features(ws: WaferSeries) -> dict[str, float]:
    """웨이퍼 1장 → {컬럼명: 값} 요약 피처.

    NaN 은 nan-safe 축약으로 무시한다(디코딩 값에 결측이 있어도 안전).
    """
    data = ws.data  # (frames, n_feat) float32
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)
    slopes = _time_slope(ws.times, data)
    p05, p95 = np.nanpercentile(data, [5, 95], axis=0)
    ranges = p95 - p05

    feats: dict[str, float] = {}
    for i, name in enumerate(ws.features):
        c = short_channel(name)
        feats[f"{c}__mean"] = float(means[i])
        feats[f"{c}__std"] = float(stds[i])
        feats[f"{c}__slope"] = float(slopes[i])
        feats[f"{c}__range"] = float(ranges[i])
    return feats


def build_feature_table(groups: list[str] | None = None,
                        limit: int | None = None) -> pd.DataFrame:
    """웨이퍼 그룹들을 순회하며 요약 피처 테이블 생성.

    Parameters
    ----------
    groups : 대상 그룹명 목록. None 이면 전체(list_wafer_groups).
    limit  : 앞에서 N장만 처리(1장 검증용). None 이면 전체.

    채널 정렬은 valid_features()(유효 27채널)로 고정 → 모든 웨이퍼 동일 열.
    """
    if groups is None:
        groups = list_wafer_groups()
    if limit is not None:
        groups = groups[:limit]

    feats = valid_features()  # 27채널, 죽은 채널 제외, 순서 고정
    rows: list[dict] = []
    for k, name in enumerate(groups, 1):
        t0 = _time.perf_counter()
        ws = read_wafer(name, features=feats)  # 지정 채널만, 순서 고정
        row = {
            "group_name": ws.group_name,
            "date": ws.date,
            "wafer_number": ws.wafer_number,
            "n_frames": ws.n_frames,
        }
        row.update(wafer_features(ws))
        rows.append(row)
        dt = _time.perf_counter() - t0
        print(f"  [{k:2d}/{len(groups)}] {name}  frames={ws.n_frames}  ({dt:.2f}s)")

    df = pd.DataFrame(rows)
    # 키 컬럼을 앞으로 정렬
    key_cols = ["group_name", "date", "wafer_number", "n_frames"]
    feat_cols = [c for c in df.columns if c not in key_cols]
    return df[key_cols + sorted(feat_cols)]


def main() -> None:
    ap = argparse.ArgumentParser(description="웨이퍼당 공정 요약 피처 생성")
    ap.add_argument("--limit", type=int, default=None,
                    help="앞에서 N장만 처리(1장 검증용)")
    ap.add_argument("--no-save", action="store_true",
                    help="parquet 저장 없이 요약만 출력(검증용)")
    args = ap.parse_args()

    ensure_output_dirs()
    print(f"유효 채널 {len(valid_features())}개 × 통계 {len(STAT_NAMES)}종 "
          f"= 피처 {len(valid_features()) * len(STAT_NAMES)}개")

    t0 = _time.perf_counter()
    df = build_feature_table(limit=args.limit)
    print(f"\n완료: {df.shape[0]} 웨이퍼 × {df.shape[1]} 열 "
          f"(총 {_time.perf_counter() - t0:.1f}s)")

    # sanity: 피처 열 결측/이상 점검
    feat_cols = [c for c in df.columns
                 if c not in ("group_name", "date", "wafer_number", "n_frames")]
    n_nan = int(df[feat_cols].isna().sum().sum())
    print(f"피처 결측(NaN) 총 {n_nan}개")
    print("\n[미리보기] 대표 채널 요약:")
    preview = [c for c in df.columns if c.startswith(
        ("SourceRFLoadPower__", "Gas1Flow__", "PlatenDcBias__"))]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df[["group_name"] + preview].head(3).to_string(index=False))

    if not args.no_save:
        df.to_parquet(PROCESS_FEATURES_PARQUET, index=False)
        print(f"\n저장: {PROCESS_FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
