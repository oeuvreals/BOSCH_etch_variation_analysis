"""프로젝트 공통 경로/상수 설정.

원본 데이터(.nc, .xlsx, .csv)는 프로젝트 밖 DATA_DIR 에 있음.
경로는 반드시 이 모듈을 통해서만 참조할 것 (하드코딩 금지).
환경변수 DATA_DIR 로 덮어쓸 수 있고, 기본값은 C:/202608_ncfiles.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── 원본 데이터 위치 ──────────────────────────────────────────────
# 환경변수 DATA_DIR 우선, 없으면 기본 경로.
DATA_DIR = Path(os.environ.get("DATA_DIR", r"C:/202608_ncfiles"))

# 원본 파일들
PROCESS_NC = DATA_DIR / "Process_data.nc"          # 공정 파라미터 31종, 5Hz
DICT_PROCESS_NC = DATA_DIR / "Dictionary_process.nc"  # uint16 → float32 코드북
DICT_OES_NC = DATA_DIR / "Dictionary_OES.nc"       # OES 디코더 (A 라인 미사용)
LOT_STATUS_XLSX = DATA_DIR / "Lot_status.xlsx"     # lot ↔ 날짜 ↔ 컨디셔닝
ETCH_9PT_CSV = DATA_DIR / "Si_Oxide_etch_9_points.csv"
ETCH_89PT_CSV = DATA_DIR / "Si_Oxide_etch_89_points.csv"

# ── 프로젝트 내부 산출물 위치 ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"   # 중간 산출물 (gitignore)
FIGURES_DIR = PROJECT_ROOT / "figures"                 # 그림 PNG, dpi=150

# A 라인 산출물
PROCESS_FEATURES_PARQUET = DATA_PROCESSED / "process_features.parquet"
MODEL_TABLE_PARQUET = DATA_PROCESSED / "model_table.parquet"  # 조인 완료 모델 입력

# B 라인 산출물 (커밋 예외로 읽기 가능)
WAFERS_9PT_CSV = DATA_PROCESSED / "wafers_9pt.csv"
WAFERS_89PT_CSV = DATA_PROCESSED / "wafers_89pt.csv"
LOT_CONDITIONS_CSV = DATA_PROCESSED / "lot_conditions.csv"

# ── 파싱 파라미터 (CLAUDE.md 규칙) ────────────────────────────────
CHUNK_FRAMES = 2000      # .nc 는 웨이퍼 단위 + 청크로만 읽는다
DOWNCAST_DTYPE = "float32"  # float64 디코딩 금지 (웨이퍼당 440MB)


def ensure_output_dirs() -> None:
    """산출물 디렉터리 생성 (없으면)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # 경로 점검용
    print(f"DATA_DIR          = {DATA_DIR}  (exists={DATA_DIR.exists()})")
    for name in ["PROCESS_NC", "DICT_PROCESS_NC", "LOT_STATUS_XLSX",
                 "ETCH_9PT_CSV", "ETCH_89PT_CSV"]:
        p = globals()[name]
        print(f"  {name:18s}= {p}  (exists={p.exists()})")
    print(f"PROJECT_ROOT      = {PROJECT_ROOT}")
    print(f"DATA_PROCESSED    = {DATA_PROCESSED}")
