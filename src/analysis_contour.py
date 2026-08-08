"""5단계 - 89점 컨투어 맵: 웨이퍼 내 공간 구조와 drift 의 공간 분포.

3단계에서 89점 산포의 98.2%(sd 3.578 µm)가 "웨이퍼 내 위치(loc)" 성분임을 밝혔지만
그 성분이 공간적으로 어떤 모양인지는 답하지 못했다. 4단계에서는 drift 기울기가
9점 grid 에서 -0.1053, 89점 전체에서 -0.1189 µm/장으로 갈리는 이유가 순전히
"측정 grid 차이" 임을 보였고, "9점이 놓치는 영역이 더 빨리 drift 한다" 는 추론을 남겼다.

여기서는 그 추론을 공간적으로 확인하고, loc 성분을 반경 / 방위각 / 국소로 분해한다.

설계 결정 (사용자 확정)
    1. 그림은 griddata(cubic) 로 렌더하되, 별도로 극좌표 모델을 적합해
       반경 성분과 tilt 성분의 기여도를 숫자로 분리한다.
    2. 주 산출물은 drift 맵(좌표별 기울기). 평균 맵은 해석 기준선으로 같은
       그림의 참조 패널로만 넣는다.
    3. 방위각 비대칭은 진폭·방향을 정량화하고, lot 간 방향 일관성(Rayleigh)까지
       검정해 장비 고정 비대칭인지 웨이퍼 배치 오차인지 좁힌다.
    4. 반경 프로파일이 단조가 아니라 보울형이므로 반경은 다항식이 아니라
       링 고정효과 C(ring_id) 로 비모수 처리한다.

출력
    data/processed/contour_coord_stats.csv
    data/processed/contour_ring_stats.csv
    data/processed/contour_tilt_by_lot.csv
    data/processed/contour_grid_loss.csv
    figures/contour_drift.png
    figures/contour_asymmetry.png
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import griddata
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
DATA_DIRS = [ROOT / "datasets", ROOT / "data" / "raw"]

VALUE = "si_etch"
WAFER_R_MM = 100.0          # 200 mm 웨이퍼
GRID_N = 400                # 렌더 그리드 해상도


def find(name):
    for d in DATA_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def style():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 데이터 준비


def load_points():
    """89점 원본 + 극좌표 파생. 중심점(r=0)은 theta 가 정의되지 않아 cos/sin=0."""
    d = pd.read_csv(find("Si_Oxide_etch_89_points.csv")).dropna(subset=["experiment_key"])
    d["is_interp"] = d["postox_thickness_nan"].isna()
    d["r_mm"] = np.hypot(d["X"], d["Y"]) / 1000.0
    d["theta"] = np.arctan2(d["Y"], d["X"])

    r_key = d["r_mm"].round(2)
    rings = np.sort(r_key.unique())
    d["ring_r"] = r_key
    d["ring_id"] = r_key.map({r: i for i, r in enumerate(rings)}).astype(int)

    at_center = d["r_mm"] < 1e-9
    d["cos_t"] = np.where(at_center, 0.0, np.cos(d["theta"]))
    d["sin_t"] = np.where(at_center, 0.0, np.sin(d["theta"]))

    n_coord = d.groupby(["X", "Y"]).ngroups
    assert n_coord == 89, f"좌표가 89개가 아니다: {n_coord}"
    assert len(rings) == 15, f"반경 링이 15개가 아니다: {len(rings)}"
    return d


def load_grid9():
    """9점 계측 좌표 9개. 89점 좌표의 진부분집합임을 확인한다."""
    d9 = pd.read_csv(find("Si_Oxide_etch_9_points.csv")).dropna(subset=["experiment_key"])
    return set(zip(d9["X"], d9["Y"]))


# ---------------------------------------------------------------- 좌표별 기울기


def coord_slopes(d, value=VALUE):
    """89개 좌표 각각에서 lot 고정효과 pooled slope (4단계 pooled_slope 와 동일 형태).

    좌표당 약 90행, lot 더미 9개 + 기울기 1개 -> df 충분.
    89회 검정이므로 BH FDR 보정을 함께 싣는다.
    """
    rows = []
    for (x, y), g in d.groupby(["X", "Y"]):
        fit = smf.ols(f"{value} ~ C(lot_number) + wafer_number", data=g).fit()
        lo, hi = fit.conf_int().loc["wafer_number"]
        rows.append(
            {
                "X": x,
                "Y": y,
                "r_mm": g["r_mm"].iloc[0],
                "theta_deg": np.degrees(g["theta"].iloc[0]),
                "ring_id": g["ring_id"].iloc[0],
                "ring_r": g["ring_r"].iloc[0],
                "cos_t": g["cos_t"].iloc[0],
                "sin_t": g["sin_t"].iloc[0],
                "n": len(g),
                "n_interp": int(g["is_interp"].sum()),
                "mean": g[value].mean(),
                "sd": g[value].std(ddof=1),
                "slope": fit.params["wafer_number"],
                "se": fit.bse["wafer_number"],
                "ci_lo": lo,
                "ci_hi": hi,
                "p": fit.pvalues["wafer_number"],
            }
        )
    cs = pd.DataFrame(rows)
    cs["p_fdr"] = multipletests(cs["p"], method="fdr_bh")[1]

    # IDW 보간점 제외 시 좌표 평균이 얼마나 흔들리는지 (4단계에서 무영향 확인됨)
    clean = d[~d["is_interp"]].groupby(["X", "Y"])[value].mean()
    cs["mean_no_interp"] = [clean.get((x, y), np.nan) for x, y in zip(cs["X"], cs["Y"])]
    cs["interp_shift"] = cs["mean_no_interp"] - cs["mean"]
    return cs.sort_values(["ring_id", "theta_deg"]).reset_index(drop=True)


# ---------------------------------------------------------------- 공간 분해


def spatial_decomposition(cs, col):
    """반경(링 고정효과) / 방위각(1차 조화항) / 국소 로 공간 패턴을 쪼갠다.

    반경 프로파일이 보울형(비단조)이라 다항식 대신 C(ring_id) 로 비모수 처리한다.
    -> C(ring_id) 단독 R^2 = 반경 기여도
       cos_t+sin_t 추가 delta R^2 = 방위각(tilt) 기여도
       나머지 = 국소·랜덤
    """
    df = cs.rename(columns={col: "v"})
    m_ring = smf.ols("v ~ C(ring_id)", data=df).fit()
    m_full = smf.ols("v ~ C(ring_id) + cos_t + sin_t", data=df).fit()
    tab = sm.stats.anova_lm(m_ring, m_full)

    a, b = m_full.params["cos_t"], m_full.params["sin_t"]
    return {
        "r2_radial": m_ring.rsquared,
        "r2_full": m_full.rsquared,
        "d_r2_tilt": m_full.rsquared - m_ring.rsquared,
        "r2_local": 1 - m_full.rsquared,
        "tilt_F": tab["F"].iloc[1],
        "tilt_p": tab["Pr(>F)"].iloc[1],
        "a": a,
        "b": b,
        "amp": float(np.hypot(a, b)),
        "direction_deg": float(np.degrees(np.arctan2(b, a))),
        "resid_ring": np.asarray(m_ring.resid),  # 반경 제거 후 = 비대칭 + 국소
    }


def tilt_by_lot(d, value=VALUE):
    """lot 마다 웨이퍼 평균 -> 89좌표에 같은 극좌표 모델 적합 -> tilt 벡터 10개."""
    rows = []
    for lot, g in d.groupby("lot_number"):
        cm = (
            g.groupby(["X", "Y"])
            .agg(v=(value, "mean"), ring_id=("ring_id", "first"),
                 cos_t=("cos_t", "first"), sin_t=("sin_t", "first"))
            .reset_index()
        )
        fit = smf.ols("v ~ C(ring_id) + cos_t + sin_t", data=cm).fit()
        a, b = fit.params["cos_t"], fit.params["sin_t"]
        rows.append(
            {
                "lot_number": lot,
                "n_wafers": g["wafer_number"].nunique(),
                "a": a,
                "b": b,
                "amp": float(np.hypot(a, b)),
                "direction_deg": float(np.degrees(np.arctan2(b, a))),
                "p_cos": fit.pvalues["cos_t"],
                "p_sin": fit.pvalues["sin_t"],
            }
        )
    return pd.DataFrame(rows)


def rayleigh(direction_deg):
    """방향 집중도 검정. n=10 이라 근사식을 쓴다 (Zar, Biostatistical Analysis).

    귀무가설: 방향이 원 위에 균일 분포 (= lot 마다 비대칭 방향이 제멋대로).
    기각되면 방향이 한쪽으로 모인다 = 장비에 고정된 비대칭.
    """
    th = np.radians(np.asarray(direction_deg, dtype=float))
    n = len(th)
    C, S = np.cos(th).sum(), np.sin(th).sum()
    Rbar = np.hypot(C, S) / n
    Z = n * Rbar ** 2
    p = np.exp(-Z) * (
        1
        + (2 * Z - Z ** 2) / (4 * n)
        - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2)
    )
    return {"n": n, "Rbar": Rbar, "Z": Z, "p": float(np.clip(p, 0, 1)),
            "mean_dir_deg": float(np.degrees(np.arctan2(S, C)))}


# ---------------------------------------------------------------- grid 손실


def grid_coverage_loss(d, grid9, value=VALUE):
    """같은 89점 데이터에서 9점 좌표만 남겼을 때 무엇을 놓치는지 정량화."""
    on9 = np.array([xy in grid9 for xy in zip(d["X"], d["Y"])])
    assert on9.sum() > 0, "89점 안에서 9점 좌표를 찾지 못했다"

    rows = []
    for label, sub in (("89점 전체", d), ("9점 grid", d[on9])):
        per_w = sub.groupby(["lot_number", "wafer_number"])[value].agg(
            mean="mean", sd="std", vmin="min", vmax="max", n="size")
        per_w["range_pct"] = 100 * (per_w["vmax"] - per_w["vmin"]) / per_w["mean"]
        wm = per_w.reset_index()
        fit = smf.ols("mean ~ C(lot_number) + wafer_number", data=wm).fit()
        lo, hi = fit.conf_int().loc["wafer_number"]
        rows.append(
            {
                "grid": label,
                "n_coords": sub.groupby(["X", "Y"]).ngroups,
                "r_max_mm": sub["r_mm"].max(),
                "level_mean": per_w["mean"].mean(),
                "within_wafer_sd": per_w["sd"].mean(),
                "nonuniformity_pct": per_w["range_pct"].mean(),
                "pooled_slope": fit.params["wafer_number"],
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 보간


def surface(cs, col, method="cubic"):
    """89점 -> 연속 면. 볼록껍질 밖과 최외곽 반경 밖은 NaN 마스킹."""
    g = np.linspace(-WAFER_R_MM, WAFER_R_MM, GRID_N)
    GX, GY = np.meshgrid(g, g)
    pts = np.column_stack([cs["X"] / 1000.0, cs["Y"] / 1000.0])
    Z = griddata(pts, cs[col].values, (GX, GY), method=method)
    Z = np.where(np.hypot(GX, GY) <= cs["r_mm"].max(), Z, np.nan)
    return GX, GY, Z


def interp_uncertainty(cs, col):
    """cubic(CloughTocher)은 오버슈트할 수 있다. linear 면과의 최대 괴리를 보고."""
    _, _, zc = surface(cs, col, "cubic")
    _, _, zl = surface(cs, col, "linear")
    ok = ~(np.isnan(zc) | np.isnan(zl))
    diff = np.abs(zc - zl)[ok]
    rng = cs[col].max() - cs[col].min()
    return {
        "max_abs_diff": float(diff.max()),
        "p99_abs_diff": float(np.percentile(diff, 99)),
        "data_range": float(rng),
        "max_pct_of_range": float(100 * diff.max() / rng),
        "cubic_overshoot": float(np.nanmax(zc) - cs[col].max()),
        "cubic_undershoot": float(cs[col].min() - np.nanmin(zc)),
    }


# ---------------------------------------------------------------- 리포트


def report():
    d = load_points()
    grid9 = load_grid9()
    assert grid9 <= set(zip(d["X"], d["Y"])), "9점 좌표가 89점의 부분집합이 아니다"

    cs = coord_slopes(d)
    ring = (
        cs.groupby("ring_r")
        .agg(n_coords=("mean", "size"), mean=("mean", "mean"), mean_sd=("mean", "std"),
             slope=("slope", "mean"), slope_sd=("slope", "std"))
        .reset_index()
        .sort_values("ring_r")
    )
    dec_mean = spatial_decomposition(cs, "mean")
    dec_slope = spatial_decomposition(cs, "slope")
    tl = tilt_by_lot(d)
    ray = rayleigh(tl["direction_deg"])
    loss = grid_coverage_loss(d, grid9)
    unc_mean = interp_uncertainty(cs, "mean")
    unc_slope = interp_uncertainty(cs, "slope")

    on9 = cs.apply(lambda r: (r["X"], r["Y"]) in grid9, axis=1)
    outer = cs[cs.ring_id == cs.ring_id.max()]

    print(f"\n{'=' * 78}")
    print("5단계 - 89점 컨투어: 공간 구조와 drift 의 공간 분포")
    print(f"  웨이퍼 {d.groupby(['lot_number', 'wafer_number']).ngroups}장 x 좌표 89개 = {len(d)}행")
    print(f"{'=' * 78}")

    print("\n[1] 반경 프로파일 - 단조가 아니라 보울(W)형")
    print(ring.round(4).to_string(index=False))
    c, e = ring["mean"].iloc[0], ring["mean"].iloc[-1]
    lo_r = ring.loc[ring["mean"].idxmin()]
    print(f"    중심 {c:.2f} -> 최소 {lo_r['mean']:.2f} (r={lo_r['ring_r']:.2f} mm)"
          f" -> 최외곽 {e:.2f} um")
    print(f"    edge roll-up : 중심 대비 {100 * (e - c) / c:+.1f}%")

    print("\n[2] 좌표별 drift 기울기")
    print(f"    89개 평균 {cs['slope'].mean():+.4f} um/장   (4단계 89점 pooled -0.1189 대조)")
    print(f"    9점 좌표 9개 평균 {cs.loc[on9, 'slope'].mean():+.4f} um/장"
          f"   (4단계 89pt->9grid -0.1053 대조)")
    print(f"    범위 {cs['slope'].min():+.4f} ~ {cs['slope'].max():+.4f}"
          f"   음수 {(cs['slope'] < 0).sum()}/89   FDR<0.05 {(cs['p_fdr'] < 0.05).sum()}/89")
    # edge = r >= 95 mm 의 두 링(좌표 20개)을 좌표수 가중 평균
    edge = ring[ring["ring_r"] >= 95.0]
    s_edge = float(np.average(edge["slope"], weights=edge["n_coords"]))
    s_ctr = float(ring["slope"].iloc[0])
    print(f"    중심(r=0) {s_ctr:+.4f} -> edge(r>=95mm, 좌표 {int(edge['n_coords'].sum())}개)"
          f" {s_edge:+.4f}  = edge 가 {s_edge / s_ctr:.2f}배 빠르게 drift")
    print(f"    링별 최대 {ring['slope'].min():+.4f} (r={ring.loc[ring['slope'].idxmin(), 'ring_r']:.2f} mm)"
          f"  -> 중심 대비 {ring['slope'].min() / s_ctr:.2f}배")

    print("\n[3] 공간 패턴 분해 (3단계 loc 성분 98.2% 를 한 겹 더 쪼갬)")
    for name, dec in (("평균 면", dec_mean), ("drift 면", dec_slope)):
        print(f"    {name:8s} 반경 {100 * dec['r2_radial']:5.1f}%"
              f" | 방위각(tilt) {100 * dec['d_r2_tilt']:4.1f}%"
              f" | 국소 {100 * dec['r2_local']:4.1f}%"
              f"   tilt F={dec['tilt_F']:.2f} p={dec['tilt_p']:.2e}")

    print("\n[4] 방위각 비대칭 - 장비 고정인가 lot 마다 랜덤인가")
    print(f"    전체 tilt 진폭 {dec_mean['amp']:.3f} um, 방향 {dec_mean['direction_deg']:+.1f} deg")
    print(f"    최외곽 링(r={outer['ring_r'].iloc[0]:.2f}) 스프레드"
          f" {outer['mean'].max() - outer['mean'].min():.2f} um"
          f"  ({outer['mean'].min():.2f} ~ {outer['mean'].max():.2f})")
    print(tl.round(4).to_string(index=False))
    verdict = ("방향 집중 -> 장비 고정 비대칭 (가스 인젝터/펌프 포트/RF 급전 등 챔버 기하)"
               if ray["p"] < 0.05 else "균일 기각 못 함 -> 방향 일관성 근거 부족")
    print(f"    Rayleigh(근사): n={ray['n']}  Rbar={ray['Rbar']:.3f}  p={ray['p']:.4f}"
          f"  평균방향 {ray['mean_dir_deg']:+.1f} deg")
    print(f"    -> {verdict}")

    print("\n[5] 9점 grid 커버리지 손실")
    print(loss.round(4).to_string(index=False))
    a89, a9 = loss.iloc[0], loss.iloc[1]
    print(f"    9점만 보면  비균일도 {100 * (1 - a9['nonuniformity_pct'] / a89['nonuniformity_pct']):.0f}%"
          f" / 웨이퍼내 SD {100 * (1 - a9['within_wafer_sd'] / a89['within_wafer_sd']):.0f}%"
          f" / drift {100 * (1 - a9['pooled_slope'] / a89['pooled_slope']):.0f}% 과소평가")
    print(f"    커버 반경 {a9['r_max_mm']:.1f} mm vs {a89['r_max_mm']:.1f} mm"
          f"  -> r>{a9['r_max_mm']:.0f} mm 미측정")

    print("\n[6] 보간 불확실성 (cubic vs linear)")
    for name, u in (("평균 면", unc_mean), ("drift 면", unc_slope)):
        print(f"    {name:8s} max|cubic-linear| {u['max_abs_diff']:.4f}"
              f" ({u['max_pct_of_range']:.1f}% of range {u['data_range']:.3f})"
              f"  오버슈트 {u['cubic_overshoot']:+.4f} / 언더슈트 {u['cubic_undershoot']:+.4f}")
    print(f"    IDW 보간점 제외 시 좌표 평균 최대 변화 {cs['interp_shift'].abs().max():.4f} um"
          f"  (보간점 보유 좌표 {(cs['n_interp'] > 0).sum()}개)")

    print("\n[7] 3단계와의 연결")
    print("    3단계 89점 loc 성분 = SS 의 98.25%, sd_equiv 3.578 um")
    print(f"    그 안에서 반경 {100 * dec_mean['r2_radial']:.1f}%"
          f" / tilt {100 * dec_mean['d_r2_tilt']:.1f}%"
          f" / 국소 {100 * dec_mean['r2_local']:.1f}%")
    print(f"    -> loc 은 랜덤 노이즈가 아니라 결정론적 반경 프로파일."
          f" sd 3.578 um 중 약 {3.578 * np.sqrt(dec_mean['r2_radial']):.2f} um 가 반경 성분")

    cs.drop(columns=["cos_t", "sin_t"]).to_csv(PROC / "contour_coord_stats.csv", index=False)
    ring.to_csv(PROC / "contour_ring_stats.csv", index=False)
    tl.to_csv(PROC / "contour_tilt_by_lot.csv", index=False)
    loss.to_csv(PROC / "contour_grid_loss.csv", index=False)

    return dict(d=d, cs=cs, ring=ring, on9=on9, grid9=grid9,
                dec_mean=dec_mean, dec_slope=dec_slope, tl=tl, ray=ray, loss=loss)


# ---------------------------------------------------------------- 그림


def _frame(ax, title):
    ax.add_patch(plt.Circle((0, 0), WAFER_R_MM, fill=False, color="k", lw=1.2))
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title, fontsize=10)


def _contour(ax, cs, col, cmap, label, title, pts_kw=None):
    GX, GY, Z = surface(cs, col, "cubic")
    cf = ax.contourf(GX, GY, Z, levels=14, cmap=cmap)
    ax.contour(GX, GY, Z, levels=14, colors="k", linewidths=0.3, alpha=0.35)
    cb = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=9)
    kw = dict(s=6, c="k", alpha=0.45, zorder=4)
    kw.update(pts_kw or {})
    ax.scatter(cs["X"] / 1000, cs["Y"] / 1000, **kw)
    _frame(ax, title)


def plot_drift(res):
    style()
    cs, ring, on9 = res["cs"], res["ring"], res["on9"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    _contour(axes[0], cs, "mean", "viridis", "Si 식각 깊이 (µm)",
             "(a) 평균 식각량 - 해석 기준선\n보울형 + edge roll-up")

    _contour(axes[1], cs, "slope", "coolwarm_r", "drift 기울기 (µm/장)",
             "(b) drift 맵 - 어디가 빨리 변하는가", pts_kw=dict(s=5, alpha=0.3))
    g9 = cs[on9]
    axes[1].scatter(g9["X"] / 1000, g9["Y"] / 1000, s=120, facecolor="none",
                    edgecolor="white", linewidth=2.2, zorder=5, label="9점 계측 위치")
    axes[1].scatter(g9["X"] / 1000, g9["Y"] / 1000, s=120, facecolor="none",
                    edgecolor="k", linewidth=0.8, zorder=6)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[2]
    ax.errorbar(ring["ring_r"], ring["mean"], yerr=ring["mean_sd"].fillna(0),
                fmt="o-", color="#4C72B0", ms=5, lw=1.6, capsize=3, label="평균 식각량")
    ax.set_xlabel("반경 r (mm)")
    ax.set_ylabel("평균 Si 식각 깊이 (µm)", color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax.twinx()
    ax2.errorbar(ring["ring_r"], ring["slope"], yerr=ring["slope_sd"].fillna(0),
                 fmt="s--", color="#C44E52", ms=5, lw=1.6, capsize=3, label="drift 기울기")
    ax2.set_ylabel("drift 기울기 (µm/장)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    r9 = sorted(cs.loc[on9, "ring_r"].unique())
    for r in r9:
        ax.axvline(r, color="grey", ls=":", lw=1.0, alpha=0.7)
    ax.axvspan(max(r9), ring["ring_r"].max() + 2, color="grey", alpha=0.13)
    ax.text(0.97, 0.06, f"9점이 못 보는 영역\nr > {max(r9):.0f} mm", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="dimgrey")
    ax.set_title("(c) 반경별 프로파일과 drift\n점선 = 9점 계측 반경", fontsize=10)
    h = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=h, frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("89점 컨투어 - 식각량 공간 분포와 drift 의 반경 의존성", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "contour_drift.png", dpi=150)
    plt.close(fig)


def plot_asymmetry(res):
    style()
    cs, dec, tl, ray = res["cs"], res["dec_mean"], res["tl"], res["ray"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    ax = axes[0]
    rcs = cs.copy()
    rcs["resid"] = dec["resid_ring"]
    m = np.abs(rcs["resid"]).max()
    lv = np.linspace(-m, m, 15)
    GX, GY, Z = surface(rcs, "resid", "cubic")
    cf = ax.contourf(GX, GY, Z, levels=lv, cmap="RdBu_r", extend="both")
    ax.contour(GX, GY, Z, levels=lv, colors="k", linewidths=0.3, alpha=0.3)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("반경 성분 제거 후 잔차 (µm)", fontsize=9)
    ax.scatter(rcs["X"] / 1000, rcs["Y"] / 1000, s=6, c="k", alpha=0.4, zorder=4)
    th = np.radians(dec["direction_deg"])
    ax.arrow(0, 0, 80 * np.cos(th), 80 * np.sin(th), width=2.5, head_width=9,
             color="k", zorder=6, length_includes_head=True)
    ax.text(0.03, 0.03,
            f"tilt 진폭 {dec['amp']:.3f} µm\n방향 {dec['direction_deg']:+.0f} deg\n"
            f"설명력 {100 * dec['d_r2_tilt']:.1f}%   p={dec['tilt_p']:.1e}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(fc="white", alpha=0.8, ec="none"))
    _frame(ax, "(a) 반경 성분을 뺀 잔차 = 방위각 비대칭")

    ax = axes[1]
    lim = tl["amp"].max() * 1.4
    for rr in np.linspace(lim / 3, lim, 3):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, color="lightgrey", lw=0.8, zorder=0))
    cmap = plt.get_cmap("tab10")
    n = len(tl)
    for i, (_, r) in enumerate(tl.iterrows()):
        ax.arrow(0, 0, r["a"], r["b"], width=lim * 0.006, head_width=lim * 0.035,
                 color=cmap(i % 10), length_includes_head=True, zorder=3)
        # 10개 벡터가 거의 겹치므로 라벨을 같은 방향선 위 서로 다른 반경에 흩뿌린다
        f = 0.30 + 0.062 * i
        ax.annotate(f"lot {int(r['lot_number'])}", (r["a"] * f, r["b"] * f), fontsize=7.5,
                    color=cmap(i % 10), ha="left", va="center",
                    xytext=(6 + (i % 2) * 26, 0), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=cmap(i % 10), alpha=0.6))
    ax.text(0.03, 0.97, f"방향 범위 {tl['direction_deg'].min():+.1f} ~ "
                        f"{tl['direction_deg'].max():+.1f} deg\n"
                        f"진폭 {tl['amp'].min():.2f} ~ {tl['amp'].max():.2f} um",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", alpha=0.85, ec="lightgrey"))
    ax.arrow(0, 0, tl["a"].mean(), tl["b"].mean(), width=lim * 0.014, head_width=lim * 0.07,
             color="k", length_includes_head=True, zorder=5, alpha=0.85)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("cos 성분 a (µm)")
    ax.set_ylabel("sin 성분 b (µm)")
    verdict = "방향 집중 (장비 고정)" if ray["p"] < 0.05 else "방향 일관성 근거 부족"
    ax.set_title(f"(b) lot 별 tilt 벡터 - {verdict}\n"
                 f"Rayleigh Rbar={ray['Rbar']:.3f}  p={ray['p']:.3f}"
                 f"  평균방향 {ray['mean_dir_deg']:+.0f} deg", fontsize=10)

    fig.suptitle("방위각 비대칭의 크기·방향과 lot 간 재현성", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "contour_asymmetry.png", dpi=150)
    plt.close(fig)


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = report()
    plot_drift(res)
    plot_asymmetry(res)
    print(f"\n저장: {FIG / 'contour_drift.png'}")
    print(f"저장: {FIG / 'contour_asymmetry.png'}")


if __name__ == "__main__":
    main()
