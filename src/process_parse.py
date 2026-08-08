"""Process_data.nc 웨이퍼 단위 파싱.

CLAUDE.md 규칙 준수:
- 통째로 메모리에 올리지 않는다. 웨이퍼(그룹) 단위 + 청크(2000프레임)로 읽는다.
- uint16 코드는 Dictionary_process.nc 의 평면 코드북으로 디코딩 (table[codes]).
- float32 다운캐스트 (float64 금지, 웨이퍼당 440MB).
- Dataset 은 set_auto_mask(False) 로 연다.
- g["data"][:] 처럼 전체 슬라이스 금지 → 청크로 읽는다.

그룹명 형식: "Day_2024_07_02_Wafer_01" → (date="2024-07-02", wafer=1).

이 모듈은 "디코딩된 원시 시계열"까지만 책임진다.
요약 피처 산출은 src/process_features.py, date→lot 조인은 이후 단계에서 처리.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from netCDF4 import Dataset

from config import CHUNK_FRAMES, DICT_PROCESS_NC, DOWNCAST_DTYPE, PROCESS_NC

_GROUP_RE = re.compile(r"^Day_(\d{4})_(\d{2})_(\d{2})_Wafer_(\d{2})$")

# 96장 전수 스캔 결과 (2026-08-08): 전 웨이퍼에서 std=0 인 죽은 채널.
# 피처 엔지니어링 입력에서 제외.
DEAD_CHANNELS = [
    "Stat3_Etch_MV_Gas3Flow",
    "Stat3_Etch_MV_Gas8Flow",
    "Stat3_Etch_MV_SourceRF2LoadPower",
    "Stat3_Etch_MV_SourceRF2ReflectedPower",
]


@dataclass
class WaferSeries:
    """웨이퍼 1장의 디코딩된 공정 시계열."""
    group_name: str
    date: str            # "YYYY-MM-DD"
    wafer_number: int
    times: np.ndarray    # (frames,) 상대 초, float64
    features: list[str]  # (n_feat,) 채널 이름
    data: np.ndarray     # (frames, n_feat) float32, 물리값

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]


def parse_group_name(name: str) -> tuple[str, int]:
    """'Day_2024_07_02_Wafer_01' → ('2024-07-02', 1)."""
    m = _GROUP_RE.match(name)
    if not m:
        raise ValueError(f"예상치 못한 그룹명 형식: {name!r}")
    y, mo, d, w = m.groups()
    return f"{y}-{mo}-{d}", int(w)


@lru_cache(maxsize=1)
def load_codebook() -> np.ndarray:
    """Dictionary_process.nc 의 평면 코드북 (float32, shape (N,)).

    각 uint16 코드는 이 배열의 인덱스. 디코딩 = codebook[codes].
    lru_cache 로 1회만 로드.
    """
    with Dataset(DICT_PROCESS_NC, "r") as dd:
        dd.set_auto_mask(False)
        table = dd.variables["data"][:].astype(DOWNCAST_DTYPE, copy=False)
    return table


def list_wafer_groups() -> list[str]:
    """Process_data.nc 의 웨이퍼 그룹명 목록 (정렬됨)."""
    with Dataset(PROCESS_NC, "r") as ds:
        ds.set_auto_mask(False)
        return sorted(ds.groups.keys())


def read_wafer(group_name: str, features: list[str] | None = None) -> WaferSeries:
    """웨이퍼 1장을 청크로 읽어 디코딩.

    Parameters
    ----------
    group_name : "Day_2024_07_02_Wafer_01"
    features : 지정 시 해당 채널만, 이 순서대로 반환 (공통 31채널 정렬용).
               None 이면 그룹에 저장된 순서 그대로.
    """
    date, wafer = parse_group_name(group_name)
    codebook = load_codebook()

    with Dataset(PROCESS_NC, "r") as ds:
        ds.set_auto_mask(False)
        g = ds.groups[group_name]
        all_feats = [str(f) for f in g.variables["feature"][:]]
        times = g.variables["times"][:].astype("float64", copy=False)

        data_var = g.variables["data"]        # (frames, n_feat) uint16
        n_frames, n_all = data_var.shape

        # 요청 채널 → 열 인덱스 (없으면 전체)
        if features is None:
            col_idx = np.arange(n_all)
            out_feats = all_feats
        else:
            name_to_col = {name: i for i, name in enumerate(all_feats)}
            missing = [f for f in features if f not in name_to_col]
            if missing:
                raise KeyError(f"{group_name}: 요청 채널 없음 {missing}")
            col_idx = np.array([name_to_col[f] for f in features])
            out_feats = list(features)

        # 청크 단위로 읽고 디코딩 (전체 슬라이스 금지)
        out = np.empty((n_frames, col_idx.size), dtype=DOWNCAST_DTYPE)
        for start in range(0, n_frames, CHUNK_FRAMES):
            stop = min(start + CHUNK_FRAMES, n_frames)
            codes = data_var[start:stop, :][:, col_idx]   # uint16 (chunk, n_sel)
            out[start:stop] = codebook[codes]              # 물리값 float32

    return WaferSeries(
        group_name=group_name, date=date, wafer_number=wafer,
        times=times, features=out_feats, data=out,
    )


def common_features(groups: list[str] | None = None) -> list[str]:
    """모든 웨이퍼에 공통으로 존재하는 채널 (정렬 기준, 44→31 공통).

    첫 그룹의 순서를 유지하며 교집합만 남긴다.
    """
    if groups is None:
        groups = list_wafer_groups()
    with Dataset(PROCESS_NC, "r") as ds:
        ds.set_auto_mask(False)
        per_group = []
        for name in groups:
            feats = [str(f) for f in ds.groups[name].variables["feature"][:]]
            per_group.append(feats)
    inter = set(per_group[0]).intersection(*map(set, per_group[1:]))
    # 첫 그룹 순서 유지
    return [f for f in per_group[0] if f in inter]


def valid_features(groups: list[str] | None = None) -> list[str]:
    """공통 채널에서 죽은 채널(DEAD_CHANNELS)을 뺀 유효 채널 목록.

    피처 엔지니어링의 기본 입력 채널 집합.
    """
    dead = set(DEAD_CHANNELS)
    return [f for f in common_features(groups) if f not in dead]


if __name__ == "__main__":
    groups = list_wafer_groups()
    print(f"웨이퍼 그룹 수: {len(groups)}")
    print(f"  첫: {groups[0]}  /  끝: {groups[-1]}")

    common = common_features(groups)
    print(f"\n공통 채널 수: {len(common)}")

    # 웨이퍼 1장 테스트 (규칙: 먼저 1장으로 검증)
    ws = read_wafer(groups[0])
    print(f"\n[테스트] {ws.group_name}  date={ws.date} wafer={ws.wafer_number}")
    print(f"  frames={ws.n_frames}, features={ws.n_features}, data.dtype={ws.data.dtype}")
    dt = np.diff(ws.times)
    print(f"  times: {ws.times[0]:.3f} → {ws.times[-1]:.3f}s  (dt~{np.median(dt):.4f}s, "
          f"{1/np.median(dt):.2f}Hz)")

    # 물리값 sanity: 채널별 min/mean/max
    print("\n  채널별 값 범위(sanity):")
    for i, name in enumerate(ws.features):
        col = ws.data[:, i]
        print(f"    {name:32s} min={np.nanmin(col):10.3f} "
              f"mean={np.nanmean(col):10.3f} max={np.nanmax(col):10.3f}")
