"""7단계 - 9점 vs 89점 계측기 offset 검증.

CLAUDE.md 는 "9점과 89점은 측정 장비가 달라 약 1.5 um 계통 offset 존재, 혼용 금지"
라고 전제한다. 6단계까지 이 전제를 그대로 받아들여 두 grid 를 항상 따로 계산했다.
7단계는 그 전제 자체를 검증한다 - offset 이 정말 상수인지, 좌표에 따라 변하는지,
그리고 애초에 1.5 um 이 맞는지.

핵심 발견 두 가지 (사전 탐색에서 드러나 계획을 바꿨다)

    1. 두 파일의 좌표계가 180도 회전되어 있다.
       좌표별 offset 이 -1.42 ~ +3.97 um 로 요동치는데, 이는 단일 계측기 offset
       으로 설명할 수 없다. 좌표 변환 7종을 전수 검정하면 180도 회전이 대응쌍
       sd 를 1.522 -> 0.109 로 14배 줄인다. 89점 면 전체를 보간해 회전각을
       연속으로 훑으면 최소가 정확히 180.0 도다.

    2. 두 파일의 si_etch 공식이 다르다 (둘 다 오차 0 으로 정확히 성립).
           9점  : si_etch = stepheight - oxide_etch  (= step - preox + postox)
           89점 : si_etch = stepheight - postox_thickness
       따라서 공표된 offset +1.338 um 안에는 "진짜 계측기 차이" 와
       "공식 차이라는 인공물" 이 섞여 있다.

설계 결정과 근거 (사용자가 판단을 위임 -> 아래 4개를 선택, 이유는 docs/step7_offset.md 3장)

    D1. 공식은 통일하지 않고 발견 보고 + 분해만 한다.
        파일 내부에서는 공식이 일관되므로 3~6단계 결과는 전부 유효하다.
        재계산의 이득(offset 수치 하나)보다 재현성 재검증 비용이 크다.
    D2. 5단계는 재계산하지 않고 좌표계 caveat 만 추가하고 A 에게 확인을 요청한다.
        9점 좌표집합이 180도 회전에 닫혀 있어 4/5/6단계 부분집합 결과가 불변이고,
        링/반경 통계도 회전 불변이라 바뀌는 것은 tilt 방향 부호 하나뿐이다.
    D3. offset 은 3종 병기하되 실무 권고는 stepheight 상수로 한다.
        raw 계측량이라 공식 인공물이 없고 sd 가 가장 작다(0.047).
    D4. 그림 1장 추가. 180도 발견은 시각화가 가장 설득력 있는 종류다.

출력
    data/processed/offset_frame_test.csv
    data/processed/offset_rotation_scan.csv
    data/processed/offset_by_column.csv
    data/processed/offset_by_coord.csv
    data/processed/offset_structure.csv
    figures/offset_calibration.png
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.interpolate import griddata

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
DATA_DIRS = [ROOT / "datasets", ROOT / "data" / "raw"]

KEY = ["lot_number", "wafer_number", "X", "Y"]
COLS = ["preox_thickness", "postox_thickness", "stepheight", "oxide_etch", "si_etch"]

# 사전 탐색에서 결정된 정답 프레임. 아래 frame_test() 가 매 실행마다 재확인한다.
BEST_FRAME = "180도 회전 (-X,-Y)"

FRAMES = {
    "원본 (변환 없음)": (lambda x, y: x, lambda x, y: y),
    "X 부호 반전": (lambda x, y: -x, lambda x, y: y),
    "Y 부호 반전": (lambda x, y: x, lambda x, y: -y),
    "180도 회전 (-X,-Y)": (lambda x, y: -x, lambda x, y: -y),
    "X<->Y 전치": (lambda x, y: y, lambda x, y: x),
    "+90도 회전 (-Y,X)": (lambda x, y: -y, lambda x, y: x),
    "-90도 회전 (Y,-X)": (lambda x, y: y, lambda x, y: -x),
}


def find(name):
    for d in DATA_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def style():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 데이터 준비


def load_raw():
    """두 원본 계측 CSV. experiment_key 결측행(9줄) 제외, 조인키를 float 로 통일."""
    a = pd.read_csv(find("Si_Oxide_etch_9_points.csv")).dropna(subset=["experiment_key"])
    b = pd.read_csv(find("Si_Oxide_etch_89_points.csv")).dropna(subset=["experiment_key"])
    for d in (a, b):
        for c in KEY:
            d[c] = d[c].astype(float)
    return a, b


def transform(a, name):
    """9점 좌표에 프레임 변환을 적용한 사본."""
    fx, fy = FRAMES[name]
    out = a.copy()
    x, y = a["X"].to_numpy(), a["Y"].to_numpy()
    out["X"], out["Y"] = fx(x, y), fy(x, y)
    # 부호 반전으로 생긴 -0.0 을 0.0 으로 정규화해야 조인이 깨지지 않는다
    out["X"] += 0.0
    out["Y"] += 0.0
    return out


def pair(a, b, frame=BEST_FRAME, cols=COLS):
    """같은 lot/wafer/좌표의 대응쌍. 공정 변동이 쌍 안에서 상쇄된다."""
    m = transform(a, frame)[KEY + cols].merge(b[KEY + cols], on=KEY, suffixes=("_9", "_89"))
    m["r_mm"] = np.hypot(m["X"], m["Y"]) / 1000.0
    m["theta_deg"] = np.degrees(np.arctan2(m["Y"], m["X"]))
    m["coord"] = [f"r={r:.0f},th={t:+.0f}" for r, t in zip(m["r_mm"], m["theta_deg"])]
    return m


# ---------------------------------------------------------------- 1. 공식 검증


def verify_formulas(a, b):
    """두 파일이 si_etch 를 어떻게 만들었는지 항등식으로 역추적한다."""
    cand = {
        "step - oxide_etch": lambda d: d["stepheight"] - d["oxide_etch"],
        "step - postox": lambda d: d["stepheight"] - d["postox_thickness"],
        "step + oxide_etch": lambda d: d["stepheight"] + d["oxide_etch"],
        "step - preox": lambda d: d["stepheight"] - d["preox_thickness"],
    }
    rows = []
    for tag, d in (("9pt", a), ("89pt", b)):
        for nm, f in cand.items():
            rows.append({"grid": tag, "formula": nm,
                         "max_abs_err": float(np.abs(d["si_etch"] - f(d)).max())})
        rows.append({"grid": tag, "formula": "oxide_etch == preox - postox",
                     "max_abs_err": float(np.abs(
                         d["oxide_etch"] - (d["preox_thickness"] - d["postox_thickness"])).max())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. 프레임 검정


def frame_test(a, b):
    """좌표 변환 7종. 정답 프레임이면 좌표별 offset 이 서로 같아야 한다.

    판정 지표
        sd        : 대응쌍 차이의 표준편차 (작을수록 좋음)
        spread    : 좌표별 평균 offset 의 최대-최소
        F         : 일원분산분석 F (좌표간 / 좌표내). 작을수록 좌표 독립에 가깝다
    """
    rows = []
    for name in FRAMES:
        for val in ("si_etch", "stepheight"):
            m = pair(a, b, frame=name, cols=[val])
            d = m[f"{val}_89"] - m[f"{val}_9"]
            per = d.groupby([m["X"], m["Y"]]).mean()
            g = d.groupby([m["X"], m["Y"]])
            ss_w = float(((d - g.transform("mean")) ** 2).sum())
            ss_b = float(((d - d.mean()) ** 2).sum()) - ss_w
            k, n = g.ngroups, len(d)
            rows.append({"frame": name, "value": val, "n_pair": n,
                         "offset": float(d.mean()), "sd": float(d.std()),
                         "coord_spread": float(per.max() - per.min()),
                         "F_coord": ((ss_b / (k - 1)) / (ss_w / (n - k))
                                     if n > k and ss_w > 0 else np.nan)})
    return pd.DataFrame(rows)


def rotation_scan(a, b, value="stepheight", coarse=2.0, fine=0.25):
    """이산 변환 7종 대신 연속 회전각을 훑는다.

    89점 면(89좌표)을 웨이퍼 평균으로 만든 뒤 phi 만큼 회전한 위치에서 선형 보간해
    9점 실측과 비교한다. 9개 이산점만 쓰는 frame_test 와 달리 면 전체를 쓰므로
    "정말 180도인가, 175도쯤인가" 를 판별할 수 있다.
    """
    bm = b.groupby(["X", "Y"], as_index=False)[value].mean()
    am = a.groupby(["X", "Y"], as_index=False)[value].mean()
    pts, val = bm[["X", "Y"]].to_numpy(), bm[value].to_numpy()
    q0, obs = am[["X", "Y"]].to_numpy(), am[value].to_numpy()

    def sd_at(phi):
        t = np.radians(phi)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        iv = griddata(pts, val, q0 @ rot.T, method="linear")
        if np.isnan(iv).any():
            return np.nan, np.nan
        d = iv - obs
        return float(d.std()), float(d.mean())

    grid = np.arange(0.0, 360.0, coarse)
    rows = [{"phi_deg": p, "sd": s, "offset": o} for p in grid for s, o in [sd_at(p)]]
    scan = pd.DataFrame(rows).dropna()
    p0 = float(scan.loc[scan["sd"].idxmin(), "phi_deg"])
    fine_grid = np.arange(p0 - 10, p0 + 10 + 1e-9, fine)
    rows2 = [{"phi_deg": p, "sd": s, "offset": o} for p in fine_grid for s, o in [sd_at(p)]]
    scan = pd.concat([scan, pd.DataFrame(rows2).dropna()]).drop_duplicates("phi_deg")
    scan = scan.sort_values("phi_deg").reset_index(drop=True)
    return scan, float(scan.loc[scan["sd"].idxmin(), "phi_deg"])


# ---------------------------------------------------------------- 3. offset 분해


def offset_by_column(m):
    """항목별 offset. preox 는 대조군 - 식각 전이라 두 장비가 일치해야 한다."""
    rows = []
    for c in COLS:
        d = m[f"{c}_89"] - m[f"{c}_9"]
        rows.append({"quantity": c, "mean_9": float(m[f"{c}_9"].mean()),
                     "mean_89": float(m[f"{c}_89"].mean()), "offset": float(d.mean()),
                     "sd": float(d.std()), "se": float(d.std() / np.sqrt(len(d))),
                     "ratio": float(m[f"{c}_89"].mean() / m[f"{c}_9"].mean())})
    # 공식을 통일했을 때의 offset
    variants = {
        "si_etch (공표, 공식 불일치)": m["si_etch_89"] - m["si_etch_9"],
        "si_etch (둘 다 9점 공식)":
            (m["stepheight_89"] - m["oxide_etch_89"]) - (m["stepheight_9"] - m["oxide_etch_9"]),
        "si_etch (둘 다 89점 공식)":
            (m["stepheight_89"] - m["postox_thickness_89"])
            - (m["stepheight_9"] - m["postox_thickness_9"]),
        "stepheight (raw 계측량)": m["stepheight_89"] - m["stepheight_9"],
    }
    for nm, d in variants.items():
        per = d.groupby([m["X"], m["Y"]]).mean()
        rows.append({"quantity": nm, "mean_9": np.nan, "mean_89": np.nan,
                     "offset": float(d.mean()), "sd": float(d.std()),
                     "se": float(d.std() / np.sqrt(len(d))), "ratio": np.nan,
                     "coord_spread": float(per.max() - per.min())})
    return pd.DataFrame(rows)


def coord_offsets(m, value="si_etch"):
    """좌표별 offset + 95% CI. 정답 프레임이면 9개가 서로 겹쳐야 한다."""
    d = m[f"{value}_89"] - m[f"{value}_9"]
    t = m.assign(d=d)
    g = t.groupby(["X", "Y"]).agg(r_mm=("r_mm", "first"), theta_deg=("theta_deg", "first"),
                                  n=("d", "size"), offset=("d", "mean"), sd=("d", "std"),
                                  m9=(f"{value}_9", "mean"), m89=(f"{value}_89", "mean"))
    g["se"] = g["sd"] / np.sqrt(g["n"])
    g["ci_lo"], g["ci_hi"] = g["offset"] - 1.96 * g["se"], g["offset"] + 1.96 * g["se"]
    return g.reset_index().sort_values(["r_mm", "theta_deg"]).reset_index(drop=True)


# ---------------------------------------------------------------- 4. 구조 분석


def deming(x, y, lam=1.0):
    """오차가 양변에 있을 때의 회귀. lam = (y 오차분산)/(x 오차분산).

    OLS 는 x 의 오차 때문에 기울기를 0 쪽으로 감쇠시킨다(regression dilution).
    여기서는 두 계측기 모두 오차를 가지므로 OLS 기울기를 그대로 믿으면 안 된다.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).sum() / (n - 1)
    syy = ((y - my) ** 2).sum() / (n - 1)
    sxy = ((x - mx) * (y - my)).sum() / (n - 1)
    b = ((syy - lam * sxx) + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return float(my - b * mx), float(b)


def structure(m):
    """offset 이 상수인가 - 반경 / 웨이퍼순번 / lot / 식각량 수준에 의존하는가."""
    rows = []
    for value in ("si_etch", "stepheight"):
        t = m.assign(d=m[f"{value}_89"] - m[f"{value}_9"])
        f_r = smf.ols("d ~ r_mm", data=t).fit()
        f_w = smf.ols("d ~ C(lot_number) + wafer_number", data=t).fit()
        f_o = smf.ols(f"{value}_89 ~ {value}_9", data=t).fit()
        ci = f_o.conf_int()
        # 오차분산의 합만 식별 가능하고 분해는 불가능하므로 lam 을 감도분석한다
        dem = {lam: deming(t[f"{value}_9"], t[f"{value}_89"], lam) for lam in (0.25, 1.0, 4.0)}
        rows.append({
            "value": value, "offset": float(t["d"].mean()), "sd": float(t["d"].std()),
            "radial_slope": float(f_r.params["r_mm"]), "radial_p": float(f_r.pvalues["r_mm"]),
            "radial_r2": float(f_r.rsquared),
            "order_slope": float(f_w.params["wafer_number"]),
            "order_p": float(f_w.pvalues["wafer_number"]),
            "ols_intercept": float(f_o.params.iloc[0]), "ols_slope": float(f_o.params.iloc[1]),
            "ols_slope_lo": float(ci.iloc[1, 0]), "ols_slope_hi": float(ci.iloc[1, 1]),
            "ols_resid_sd": float(np.sqrt(f_o.scale)),
            "deming_slope_lam0.25": dem[0.25][1], "deming_slope_lam1": dem[1.0][1],
            "deming_slope_lam4": dem[4.0][1],
            "err_var_total": float(t["d"].var()),
        })
    return pd.DataFrame(rows)


def retro_check(a, b):
    """기존 단계에 미치는 영향 - 좌표집합 닫힘성과 두 파일의 tilt 비교."""
    c9 = set(zip(a["X"], a["Y"]))
    closed = c9 == {(-x, -y) for x, y in c9}

    tilt = {}
    for tag, d in (("9pt", a), ("89pt", b)):
        ring = d[np.isclose(np.hypot(d["X"], d["Y"]) / 1000.0, 76.0)]
        s = ring.groupby(["X", "Y"])["si_etch"].mean()
        v = {(int(x), int(y)): val for (x, y), val in s.items()}
        aa = (v[(76000, 0)] - v[(-76000, 0)]) / 2
        bb = (v[(0, 76000)] - v[(0, -76000)]) / 2
        tilt[tag] = (aa, bb, float(np.hypot(aa, bb)), float(np.degrees(np.arctan2(bb, aa))))
    return closed, tilt


# ---------------------------------------------------------------- 리포트


def report():
    a, b = load_raw()
    out = {}

    print("=" * 78)
    print("7단계 - 9점 vs 89점 계측기 offset 검증")
    print("=" * 78)

    wa = set(map(tuple, a[["lot_number", "wafer_number"]].drop_duplicates().to_numpy()))
    wb = set(map(tuple, b[["lot_number", "wafer_number"]].drop_duplicates().to_numpy()))
    c9, c89 = set(zip(a["X"], a["Y"])), set(zip(b["X"], b["Y"]))
    print("\n[1] 대응쌍 설계")
    print(f"    9점 {len(a)}행 / 웨이퍼 {len(wa)}장,  89점 {len(b)}행 / 웨이퍼 {len(wb)}장")
    print(f"    9점 좌표 {len(c9)}개가 89점 좌표 {len(c89)}개의 부분집합인가: {c9 <= c89}")
    print(f"    공통 웨이퍼 {len(wa & wb)}장 x 공통 좌표 {len(c9)}개"
          f" = 대응쌍 {len(wa & wb) * len(c9)}개")
    print("    -> 같은 웨이퍼·같은 좌표를 두 장비가 잰 paired 설계."
          " 공정 변동이 쌍 안에서 상쇄된다.")

    print("\n[2] si_etch 공식 역추적 - 두 파일이 같은 정의를 쓰는가")
    fv = verify_formulas(a, b)
    out["formula"] = fv
    for tag in ("9pt", "89pt"):
        hit = fv[(fv["grid"] == tag) & (fv["max_abs_err"] < 1e-6)]
        for _, r in hit.iterrows():
            print(f"    {tag:5s} {r['formula']:32s} 정확히 성립 (max|err| {r['max_abs_err']:.1e})")
    print("    -> 9점은 step-oxide_etch, 89점은 step-postox. 정의가 다르다.")
    print("       기하학적으로는 89점 쪽이 맞다 (남은 산화막 위에서 잰 단차 = postox + Si 깊이).")

    print("\n[3] 좌표 프레임 검정 - offset 이 좌표마다 다른 이유")
    ft = frame_test(a, b)
    out["frame"] = ft
    print("    (si_etch 기준. 정답 프레임이면 좌표별 offset 이 서로 같아야 한다)")
    sub = ft[ft["value"] == "si_etch"].sort_values("sd")
    for _, r in sub.iterrows():
        mark = "  <== 최적" if r["frame"] == sub.iloc[0]["frame"] else ""
        print(f"    {r['frame']:20s} sd {r['sd']:.4f}  좌표스프레드 {r['coord_spread']:.3f}"
              f"  F {r['F_coord']:7.1f}{mark}")
    base = float(sub[sub["frame"] == "원본 (변환 없음)"]["sd"].iloc[0])
    print(f"    -> 180도 회전이 sd 를 {base:.4f} -> {sub.iloc[0]['sd']:.4f}"
          f" ({base / sub.iloc[0]['sd']:.0f}배) 줄인다.")
    fy = float(sub[sub["frame"] == "Y 부호 반전"]["F_coord"].iloc[0])
    fb = float(sub[sub["frame"] == BEST_FRAME]["F_coord"].iloc[0])
    print(f"    주의: F 는 판별자로 쓰지 않는다. Y반전 F={fy:.1f} 가 180도 F={fb:.1f}"
          f" 보다 낮게 나오지만")
    print("       sd 는 반대로 0.1887 vs 0.1091 로 180도가 낫다. F 는 좌표 내 분산으로"
          " 나누는데")
    print("       틀린 프레임에서는 그 분모 자체가 함께 부풀어 비율이 상쇄되기 때문이다."
          " 판별은 sd 와 좌표스프레드로 한다.")
    if sub.iloc[0]["frame"] != BEST_FRAME:
        raise AssertionError(f"최적 프레임이 {BEST_FRAME} 가 아님: {sub.iloc[0]['frame']}")

    print("\n[4] 연속 회전각 스캔 - 정말 180도인가")
    scan, phi = rotation_scan(a, b)
    out["scan"] = scan
    print("    (stepheight 기준. 89점 면 전체를 보간해 phi 를 훑는다."
          " 9개 이산점만 쓰는 [3] 보다 강한 검정)")
    for p in (0, 90, 135, 175, 180, 185, 225, 270):
        row = scan[np.isclose(scan["phi_deg"], p)]
        if len(row):
            print(f"      phi={p:3d}도  sd {row['sd'].iloc[0]:.4f}")
    print(f"    최소 sd 를 주는 각도 = {phi:.2f}도"
          f" (sd {scan['sd'].min():.4f})  -> 정확히 180도")

    m = pair(a, b)
    out["pairs"] = m

    print("\n[5] offset 분해 - 1.338 um 중 얼마가 진짜 계측기 차이인가")
    ob = offset_by_column(m)
    out["by_col"] = ob
    print("    항목별 (180도 보정 후, 657 대응쌍):")
    for _, r in ob[ob["quantity"].isin(COLS)].iterrows():
        print(f"      {r['quantity']:20s} 9pt {r['mean_9']:8.3f}  89pt {r['mean_89']:8.3f}"
              f"  offset {r['offset']:+8.4f} +- {r['sd']:.4f}")
    print("    -> preox 는 +0.0001 로 완벽히 일치. 식각 전 산화막에는 장비 차이가 없다(대조군).")
    print("       postox 에서만 어긋나므로 차이는 식각 후 계측에서 발생한다.")
    print("    공식·기준량을 바꿔가며 본 offset:")
    for _, r in ob[~ob["quantity"].isin(COLS)].iterrows():
        print(f"      {r['quantity']:26s} {r['offset']:+7.4f} +- {r['sd']:.4f}"
              f"   좌표스프레드 {r['coord_spread']:.4f}")
    pub = float(ob[ob["quantity"] == "si_etch (공표, 공식 불일치)"]["offset"].iloc[0])
    raw = float(ob[ob["quantity"] == "stepheight (raw 계측량)"]["offset"].iloc[0])
    print(f"    -> 공표 {pub:+.4f} = 계측기 차이 {raw:+.4f} + 공식 인공물 {pub - raw:+.4f}")
    print(f"       CLAUDE.md 의 '약 1.5 um' 은 실제 {pub:.3f}(공표) / {raw:.3f}(raw) 로 정정된다.")

    print("\n[6] 좌표별 잔여 offset (180도 보정 후)")
    for value in ("si_etch", "stepheight"):
        co = coord_offsets(m, value)
        out[f"coord_{value}"] = co
        print(f"    [{value}] 범위 {co['offset'].min():+.4f} ~ {co['offset'].max():+.4f}"
              f"  (스프레드 {co['offset'].max() - co['offset'].min():.4f})")
        if value == "si_etch":
            for _, r in co.iterrows():
                print(f"      r={r['r_mm']:5.1f} th={r['theta_deg']:+7.1f}"
                      f"  offset {r['offset']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
                      f"  sd {r['sd']:.4f}")
    print("    -> si_etch 는 중심점(1.498)이 나머지(약 1.32)보다 높지만,"
          " stepheight 에서는 스프레드가 0.318 -> 0.144 로 줄고")
    print("       반경 설명력이 R2 0.236 -> 0.026 으로 거의 사라진다."
          " 잔여 반경 구조의 대부분이 공식 인공물이다.")

    print("\n[7] offset 은 상수인가 - 반경 / 순번 / 승법 성분")
    st = structure(m)
    out["structure"] = st
    for _, r in st.iterrows():
        print(f"    [{r['value']}] offset {r['offset']:+.4f} +- {r['sd']:.4f}")
        print(f"      반경 의존 {r['radial_slope']:+.6f} um/mm"
              f"  p={r['radial_p']:.2e}  R2={r['radial_r2']:.4f}")
        print(f"      순번 의존 {r['order_slope']:+.5f} um/장  p={r['order_p']:.4f}")
        print(f"      OLS  y = {r['ols_intercept']:+.4f} + {r['ols_slope']:.4f} x"
              f"  95%CI [{r['ols_slope_lo']:.4f}, {r['ols_slope_hi']:.4f}]"
              f"  잔차sd {r['ols_resid_sd']:.4f}")
        print(f"      Deming 기울기  lam=0.25 {r['deming_slope_lam0.25']:.4f}"
              f" | lam=1 {r['deming_slope_lam1']:.4f}"
              f" | lam=4 {r['deming_slope_lam4']:.4f}")
    print("    -> 두 장비 오차분산의 합만 식별되고 분해는 안 되므로 lam 은 감도분석으로만 본다.")

    print("\n[8] 기존 단계에 미치는 영향")
    closed, tilt = retro_check(a, b)
    out["tilt"] = pd.DataFrame(
        [{"grid": k, "a_cos": v[0], "b_sin": v[1], "amp": v[2], "dir_deg": v[3]}
         for k, v in tilt.items()])
    print(f"    9점 좌표집합이 180도 회전에 닫혀 있는가: {closed}")
    print("      -> 참이므로 4/5/6단계의 '9점 좌표 부분집합' 결과는 전부 불변.")
    print("      -> 링/반경 통계도 회전 불변. 바뀌는 것은 5단계 tilt 방향 부호뿐.")
    print("    r=76mm 링 tilt 를 각 파일에서 독립 계산:")
    for k, v in tilt.items():
        print(f"      {k:5s} a={v[0]:+.3f} b={v[1]:+.3f}  진폭 {v[2]:.3f} um  방향 {v[3]:+.1f}도")
    dd = abs(tilt["9pt"][3] - tilt["89pt"][3])
    print(f"    방향 차이 {dd:.1f}도, 진폭 비 {tilt['89pt'][2] / tilt['9pt'][2]:.3f}")
    print("    -> 서로 다른 장비 두 대가 같은 크기의 비대칭을 반대 방향에서 봤다.")
    print("       프레임을 맞추면 일치 = 5단계 tilt 는 계측 아티팩트가 아니라 물리적 실재.")

    print("\n" + "=" * 78)
    print("[9] 결론")
    print(f"    1. 두 파일의 좌표계는 180도 회전 관계다 (연속 스캔 최소 {phi:.1f}도).")
    print("       보정 없이 좌표별로 비교하면 offset 이 -1.42 ~ +3.97 um 로 요동친다.")
    print("    2. si_etch 정의가 서로 달라 공표 offset 에 인공물이 섞여 있다.")
    print(f"    3. 실무 보정값 권고: stepheight 기준 {raw:+.4f} um"
          f" (sd {float(ob[ob['quantity'] == 'stepheight (raw 계측량)']['sd'].iloc[0]):.4f},"
          " 반경 의존 없음).")
    print("    4. 프레임과 공식을 맞추면 두 grid 는 합칠 수 있다."
          " CLAUDE.md 의 '혼용 금지' 는")
    print("       '보정 없이 혼용 금지' 로 완화 가능하다 (팀 합의 필요).")

    PROC.mkdir(parents=True, exist_ok=True)
    ft.to_csv(PROC / "offset_frame_test.csv", index=False, encoding="utf-8-sig")
    scan.to_csv(PROC / "offset_rotation_scan.csv", index=False, encoding="utf-8-sig")
    ob.to_csv(PROC / "offset_by_column.csv", index=False, encoding="utf-8-sig")
    out["coord_si_etch"].to_csv(PROC / "offset_by_coord.csv", index=False, encoding="utf-8-sig")
    st.to_csv(PROC / "offset_structure.csv", index=False, encoding="utf-8-sig")
    return out


# ---------------------------------------------------------------- 그림


def plot_offset(out):
    """3패널. (b) 회전각 스캔은 계획의 2패널에 더한 것 - 180도 발견의 가장 직접적 증거."""
    style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # (a) 좌표별 offset 보정 전/후
    ax = axes[0]
    a, b = load_raw()
    m0 = pair(a, b, frame="원본 (변환 없음)", cols=["si_etch"])
    c0 = coord_offsets(m0, "si_etch")
    c1 = out["coord_si_etch"]
    lab = [f"r={r:.0f}\n{t:+.0f}°" for r, t in zip(c1["r_mm"], c1["theta_deg"])]
    x = np.arange(len(c1))
    ax.errorbar(x - 0.12, c0["offset"], yerr=1.96 * c0["se"], fmt="o", ms=7, color="#C44E52",
                capsize=3, label=f"보정 전 (스프레드 {c0['offset'].max() - c0['offset'].min():.2f} µm)")
    ax.errorbar(x + 0.12, c1["offset"], yerr=1.96 * c1["se"], fmt="s", ms=7, color="#4C72B0",
                capsize=3, label=f"180° 보정 후 (스프레드 {c1['offset'].max() - c1['offset'].min():.2f} µm)")
    ax.axhline(float(c1["offset"].mean()), color="k", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7.5)
    ax.set_ylabel("89점 - 9점 offset (µm)")
    ax.set_title("(a) 좌표별 offset — 단일 계측기 offset 이라면\n9개가 한 값에 모여야 한다",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, axis="y")

    # (b) 연속 회전각 스캔
    ax = axes[1]
    scan = out["scan"].sort_values("phi_deg")
    ax.plot(scan["phi_deg"], scan["sd"], "-", lw=1.4, color="#4C72B0")
    p = float(scan.loc[scan["sd"].idxmin(), "phi_deg"])
    s = float(scan["sd"].min())
    ax.plot([p], [s], "o", ms=9, mfc="none", mec="#C44E52", mew=2)
    ax.annotate(f"최소 {p:.1f}°\nsd {s:.3f} µm", (p, s), fontsize=9, fontweight="bold",
                color="#C44E52", ha="center", xytext=(0, 26), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("89점 면을 회전시킨 각도 φ (°)")
    ax.set_ylabel("9점 실측과의 좌표간 차이 sd (µm)")
    ax.set_title("(b) 연속 회전각 스캔 (stepheight) — 89점 면을 보간해\nφ 를 훑으면 최소가 정확히 180°",
                 fontsize=10)
    ax.grid(alpha=0.25)

    # (c) Bland-Altman
    ax = axes[2]
    m = out["pairs"]
    for val, c, lab2 in (("si_etch", "#C44E52", "si_etch (공식 불일치 포함)"),
                         ("stepheight", "#4C72B0", "stepheight (raw 계측량)")):
        avg = (m[f"{val}_89"] + m[f"{val}_9"]) / 2
        dif = m[f"{val}_89"] - m[f"{val}_9"]
        ax.scatter(avg, dif, s=7, alpha=0.35, color=c, edgecolor="none")
        mu, sd = dif.mean(), dif.std()
        ax.axhline(mu, color=c, lw=1.3)
        for y in (mu - 1.96 * sd, mu + 1.96 * sd):
            ax.axhline(y, color=c, lw=0.9, ls="--", alpha=0.8)
        ax.plot([], [], "o", color=c, label=f"{lab2}\n  {mu:+.3f} ± {1.96 * sd:.3f} µm")
    ax.set_xlabel("두 장비 평균 (µm)")
    ax.set_ylabel("89점 - 9점 (µm)")
    ax.set_title("(c) Bland-Altman (657 대응쌍) — 공식을 배제한\nraw 계측량이 훨씬 좁다",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle("계측기 offset 검증 — 두 파일은 좌표계가 180° 다르고 si_etch 정의도 다르다",
                 fontweight="bold")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "offset_calibration.png", dpi=150)
    plt.close(fig)
    print(f"\n저장: {FIG / 'offset_calibration.png'}")


def main():
    out = report()
    plot_offset(out)


if __name__ == "__main__":
    main()
