"""마스터 테이블 생성: 측정점 단위 CSV -> 웨이퍼 1장 = 1행.

출력:
    data/processed/wafers_9pt.csv
    data/processed/wafers_89pt.csv
    data/processed/lot_conditions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

# 원본 위치는 로컬마다 다름 (docs/DATA.md 는 data/raw 기준)
DATA_DIRS = [ROOT / "datasets", ROOT / "data" / "raw"]

CENTER_R_MM = 20.0
EDGE_R_MM = 90.0


def find(name):
    for d in DATA_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found in {[str(d) for d in DATA_DIRS]}")


def load_lot_conditions():
    """Lot_status.xlsx 상단 표 -> lot 별 날짜/컨디셔닝 조건."""
    raw = pd.read_excel(find("Lot_status.xlsx"), sheet_name="Tabelle1", nrows=10)
    raw.columns = [c.strip() for c in raw.columns]

    df = pd.DataFrame(
        {
            "lot_number": raw["Lot No."].str.extract(r"(\d+)").astype(int)[0],
            "date": pd.to_datetime(
                raw["Date"].str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True),
                format="mixed",
            ),
            "n_wafers_planned": raw["Wafers"].astype(int),
            "measurements": raw["Measurements"].str.replace(r"\s+", " ", regex=True).str.strip(),
            "type_raw": raw["Type"].str.strip(),
        }
    )

    parts = df["type_raw"].str.split("-", expand=True)
    df["cond_rep"] = parts[0].str.extract(r"(\d+)").astype(int)
    # "3C" = 챔버(척)만, "3C - Si" / "3C - SiO2" = 더미 웨이퍼 재질
    df["cond_type"] = parts[1].fillna("Chuck").str.strip()
    df["cond_label"] = df["cond_type"] + "_" + df["cond_rep"].astype(str) + "x"
    return df.drop(columns="type_raw")


def load_points(n_points):
    name = f"Si_Oxide_etch_{n_points}_points.csv"
    df = pd.read_csv(find(name))

    if n_points == 9:
        # experiment_key 결측 9행 = 소속 웨이퍼 미상 (규칙 6)
        orphan = df["experiment_key"].isna()
        if orphan.any():
            print(f"[9pt] experiment_key 결측 {orphan.sum()}행 제외 (웨이퍼 미상)")
            df = df[~orphan].copy()
        df[["lot_number", "wafer_number"]] = df[["lot_number", "wafer_number"]].astype(int)
    else:
        # postox_thickness_nan 이 NaN = 원 측정 실패 -> postox_thickness 는 IDW 보간값
        df["is_interpolated"] = df["postox_thickness_nan"].isna()

    df["r_mm"] = np.hypot(df["X"], df["Y"]) / 1000.0
    return df


def summarize(df, n_points):
    g = df.groupby(["lot_number", "wafer_number"], sort=True)

    out = pd.DataFrame(
        {
            "experiment_key": g["experiment_key"].first(),
            "n_points": g.size(),
            "si_etch_mean": g["si_etch"].mean(),
            "si_etch_std": g["si_etch"].std(ddof=1),
            "si_etch_min": g["si_etch"].min(),
            "si_etch_max": g["si_etch"].max(),
            "oxide_etch_mean": g["oxide_etch"].mean(),
            "oxide_etch_std": g["oxide_etch"].std(ddof=1),
            "stepheight_mean": g["stepheight"].mean(),
            "preox_mean": g["preox_thickness"].mean(),
            "postox_mean": g["postox_thickness"].mean(),
        }
    )
    out["si_etch_range"] = out["si_etch_max"] - out["si_etch_min"]
    # WIW non-uniformity: 반도체 관행상 (max-min)/(2*mean)
    out["si_nu_pct"] = 100 * out["si_etch_range"] / (2 * out["si_etch_mean"])
    out["si_cv_pct"] = 100 * out["si_etch_std"] / out["si_etch_mean"]
    out["selectivity"] = out["si_etch_mean"] / out["oxide_etch_mean"]

    if n_points == 89:
        center = df[df["r_mm"] <= CENTER_R_MM].groupby(["lot_number", "wafer_number"])["si_etch"].mean()
        edge = df[df["r_mm"] >= EDGE_R_MM].groupby(["lot_number", "wafer_number"])["si_etch"].mean()
        out["si_etch_center"] = center
        out["si_etch_edge"] = edge
        out["center_to_edge"] = edge - center
        out["n_interpolated"] = g["is_interpolated"].sum()

    return out.reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lots = load_lot_conditions()
    lots.to_csv(OUT / "lot_conditions.csv", index=False)
    print(lots.to_string(index=False))

    for n in (9, 89):
        pts = load_points(n)
        wafers = summarize(pts, n).merge(lots, on="lot_number", how="left")
        wafers.insert(2, "date", wafers.pop("date"))
        path = OUT / f"wafers_{n}pt.csv"
        wafers.to_csv(path, index=False)

        print(f"\n[{n}pt] {len(wafers)} 웨이퍼 -> {path.relative_to(ROOT)}")
        have = wafers.groupby("lot_number")["wafer_number"].apply(set)
        for lot, ws in have.items():
            planned = int(lots.loc[lots.lot_number == lot, "n_wafers_planned"].iloc[0])
            missing = sorted(set(range(1, planned + 1)) - ws)
            if missing:
                print(f"  lot {lot}: 웨이퍼 {missing} 없음")
        for lot in sorted(set(lots.lot_number) - set(wafers.lot_number)):
            print(f"  lot {lot}: 전체 없음")


if __name__ == "__main__":
    main()
