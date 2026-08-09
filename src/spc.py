"""6단계 - 관리도(SPC)와 공정능력(Cp/Cpk).

4단계에서 순번 drift 를 -0.119 um/장으로, 5단계에서 웨이퍼 내 산포의 93.9% 가
결정론적 반경 프로파일임을 밝혔다. 6단계는 그 둘을 품질관리 언어로 번역한다.

핵심 질문
    1. 표준 관리도로 이 chamber drift 를 감지할 수 있는가?
    2. 감지 못 한다면 어떤 관리도가 감지하는가?
    3. 이 공정의 공정능력은 얼마인가?
    4. 능력을 깎아먹는 주범은 drift 인가 웨이퍼 내 불균일인가?

설계 결정 (사용자 확정)
    1. 규격한계가 데이터에 없으므로 임의 규격을 발명하지 않는다. 주 산출물은
       "공차 폭 vs Cpk" 민감도 곡선 + 목표 Cpk 달성에 필요한 공차 역산.
       +-3% / +-5% 두 대표값은 참고 표로만 병기한다.
    2. 관리도 3종 병기 - 원자료 I-MR(표준이 실패함을 보임) / 잔차 I-MR(특수원인
       분리) / EWMA(drift 조기감지).
    3. 특수원인은 표시 후 관리한계 산정에서 제외하고 재산정한다. 제외 전/후
       한계를 함께 보고해 영향도를 공개한다.

특수원인 2건 (근거 있는 제외, 데이터 준설 아님)
    lot 8 wafer 8 : A가 Process_data.nc 에서 규명. HeliumBPPressure 하락 /
                    Gas1Flow 상승 / moriInnerCurrent 하락, 런 길이 18% 증가
                    (3835 vs ~3246 프레임). 실제 공정 교란.
    lot 6 wafer 7 : 89점 비균일도 25.94% (나머지 87장 평균 13.27, sd 0.279
                    -> z=+45). 전체 sd 1.38 로 재면 z=+9.1 로 보이는데, 그 sd
                    자체가 이 한 점 때문에 부풀려진 값이라 과소평가다.
                    평균 잔차로는 z=-2.76 에 그쳐 평균 관리도로는 놓친다.

출력
    data/processed/spc_control_limits.csv
    data/processed/spc_violations.csv
    data/processed/spc_capability.csv
    data/processed/spc_tolerance_curve.csv
    figures/spc_control_charts.png
    figures/spc_capability.png
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
DATA_DIRS = [ROOT / "datasets", ROOT / "data" / "raw"]

D2 = 1.128          # 이동범위 n=2 의 d2 상수
EWMA_LAM = 0.2      # EWMA 가중 (drift 감지 표준값)
EWMA_L = 3.0        # EWMA 관리한계 폭
RAW_FILES = {"9pt": "Si_Oxide_etch_9_points.csv", "89pt": "Si_Oxide_etch_89_points.csv"}

# (lot_number, wafer_number) -> 제외 근거
SPECIAL = {
    (8, 8): "A: 공정신호 교란(He압/Gas1/moriI + 런길이 +18%)",
    (6, 7): "B: 웨이퍼내 비균일도 25.94% (나머지 87장 13.27+-0.28, z=+45)",
}


def find(name):
    for d in DATA_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def style():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 데이터


def load_wafers(tag):
    """웨이퍼 1행 테이블 + 실행 순번 + 특수원인 플래그."""
    w = pd.read_csv(PROC / f"wafers_{tag}.csv").sort_values(["lot_number", "wafer_number"])
    w = w.reset_index(drop=True)
    w["run"] = np.arange(1, len(w) + 1)
    w["special"] = [
        SPECIAL.get((l, n), "") for l, n in zip(w["lot_number"], w["wafer_number"])
    ]
    w["is_special"] = w["special"] != ""
    return w


def load_points(tag):
    """측정점 원자료 (소자 규격 관점의 공정능력용)."""
    d = pd.read_csv(find(RAW_FILES[tag])).dropna(subset=["experiment_key"])
    return d


# ---------------------------------------------------------------- 관리도 코어


def imr_limits(x, keep=None):
    """개별치-이동범위 관리한계.

    sigma 는 표본표준편차가 아니라 MRbar/d2 를 쓴다. 계통 변동(drift)이 있을 때
    표본 sd 는 그 변동까지 삼켜 관리한계를 부풀리므로, 인접 관측 차이만 보는
    이동범위가 '단기(short-term) 산포' 추정에 맞다.
    """
    x = np.asarray(x, dtype=float)
    m = np.ones(len(x), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    xb = x[m]
    mr = np.abs(np.diff(xb))
    sigma = mr.mean() / D2
    center = xb.mean()
    return {
        "center": center,
        "sigma": sigma,
        "lcl": center - 3 * sigma,
        "ucl": center + 3 * sigma,
        "mrbar": mr.mean(),
        "mr_ucl": 3.267 * mr.mean(),
        "n_used": int(m.sum()),
    }


def western_electric(w, col, lim):
    """WE 런 규칙. 규칙 2·3(연속성)은 lot 경계를 넘지 않도록 lot 안에서만 본다.

    lot 마다 챔버 컨디셔닝으로 상태가 리셋되므로 lot 을 가로지르는 연속 판정은
    물리적 의미가 없다.
    """
    x = w[col].to_numpy(dtype=float)
    c, s = lim["center"], lim["sigma"]
    z = (x - c) / s
    out = []

    for i in np.where(np.abs(z) > 3)[0]:                      # 규칙 1: 전역
        out.append((i, "R1: 3시그마 밖"))

    for _, g in w.groupby("lot_number"):
        idx = g.index.to_numpy()
        zz = z[idx]
        side = np.sign(zz)
        run = 1
        for k in range(1, len(zz)):                            # 규칙 2
            run = run + 1 if side[k] == side[k - 1] and side[k] != 0 else 1
            if run >= 9:
                out.append((idx[k], "R2: 같은 쪽 9연속"))
        trend = 1
        for k in range(1, len(zz)):                            # 규칙 3
            d = np.sign(zz[k] - zz[k - 1])
            trend = trend + 1 if k > 1 and d == np.sign(zz[k - 1] - zz[k - 2]) and d != 0 else 2
            if trend >= 6:
                out.append((idx[k], "R3: 6연속 단조"))
        for k in range(2, len(zz)):                            # 규칙 4
            win = zz[k - 2:k + 1]
            for sgn in (1, -1):
                if (sgn * win > 2).sum() >= 2 and sgn * zz[k] > 2:
                    out.append((idx[k], "R4: 3중 2점이 2시그마 밖"))
                    break
    if not out:
        return pd.DataFrame(columns=["idx", "rule", "lot_number", "wafer_number", "value"])
    v = pd.DataFrame(out, columns=["idx", "rule"]).drop_duplicates()
    v["lot_number"] = w.loc[v["idx"], "lot_number"].to_numpy()
    v["wafer_number"] = w.loc[v["idx"], "wafer_number"].to_numpy()
    v["value"] = x[v["idx"].to_numpy()]
    return v.sort_values(["lot_number", "wafer_number"]).reset_index(drop=True)


def ewma(w, col, center, sigma, lam=EWMA_LAM, L=EWMA_L):
    """lot 마다 리셋하는 EWMA. 컨디셔닝으로 챔버가 초기화되므로 lot 이 자연 단위.

    관리한계는 초기 과도구간을 반영해 시점 의존 폭을 쓴다:
        center +- L*sigma*sqrt(lam/(2-lam) * (1-(1-lam)^(2i)))
    """
    z_all, lo_all, hi_all, sig_all = [], [], [], []
    for _, g in w.groupby("lot_number", sort=True):
        z = center
        for i in range(1, len(g) + 1):
            z = lam * g[col].iloc[i - 1] + (1 - lam) * z
            half = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
            z_all.append(z)
            lo_all.append(center - half)
            hi_all.append(center + half)
            sig_all.append(z < center - half or z > center + half)
    out = w[["lot_number", "wafer_number", "run"]].copy()
    out["ewma"] = z_all
    out["lcl"] = lo_all
    out["ucl"] = hi_all
    out["signal"] = sig_all
    return out


def first_signal_by_lot(ew):
    """lot 마다 EWMA 가 처음 경보한 웨이퍼 순번."""
    rows = []
    for lot, g in ew.groupby("lot_number"):
        s = g[g["signal"]]
        rows.append({"lot_number": lot, "n_wafers": len(g),
                     "first_signal_wafer": int(s["wafer_number"].iloc[0]) if len(s) else np.nan,
                     "n_signal": int(g["signal"].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 공정능력


def capability(x, tol_pct, target=None, sigma=None, label=""):
    """대칭 규격에서의 Cp/Cpk.

    규격을 관측 평균 중심으로 잡으면 Cp == Cpk 가 되어 치우침 정보가 사라진다.
    실제 규격이 정해지면 치우침이 Cpk 를 추가로 떨어뜨린다는 점을 문서에 명기.
    """
    x = np.asarray(x, dtype=float)
    T = x.mean() if target is None else target
    s = x.std(ddof=1) if sigma is None else sigma
    usl, lsl = T * (1 + tol_pct / 100), T * (1 - tol_pct / 100)
    cp = (usl - lsl) / (6 * s)
    cpk = min(usl - x.mean(), x.mean() - lsl) / (3 * s)
    return {"label": label, "tol_pct": tol_pct, "lsl": lsl, "usl": usl,
            "mean": x.mean(), "sigma": s, "cp": cp, "cpk": cpk,
            "ppm": 1e6 * (stats.norm.sf((usl - x.mean()) / s)
                          + stats.norm.cdf((lsl - x.mean()) / s))}


def tolerance_for(sigma, mean, cpk_target):
    """목표 Cpk 를 얻는 데 필요한 반폭 공차(um, %)."""
    half = cpk_target * 3 * sigma
    return half, 100 * half / mean


def tolerance_curve(specs, lo=0.5, hi=8.0, n=76):
    """공차 폭 vs Cpk 곡선 (규격을 발명하지 않고 답하는 방식)."""
    rows = []
    for pct in np.linspace(lo, hi, n):
        r = {"tol_pct": pct}
        for label, (mean, sigma) in specs.items():
            r[label] = (mean * pct / 100) / (3 * sigma)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 리포트


def analyze(tag):
    w = load_wafers(tag)
    pts = load_points(tag)
    keep = ~w["is_special"].to_numpy()

    lim_all = imr_limits(w["si_etch_mean"])
    lim_cln = imr_limits(w["si_etch_mean"], keep)

    fit = smf.ols("si_etch_mean ~ C(lot_number) + wafer_number", data=w[keep]).fit()
    w["resid"] = w["si_etch_mean"] - fit.predict(w)
    lim_res = imr_limits(w.loc[keep, "resid"])

    lim_nu_all = imr_limits(w["si_nu_pct"])
    lim_nu = imr_limits(w["si_nu_pct"], keep)

    we_raw = western_electric(w, "si_etch_mean", lim_cln)
    we_res = western_electric(w, "resid", lim_res)
    ew = ewma(w, "si_etch_mean", lim_cln["center"], lim_cln["sigma"])

    sd_w = w["si_etch_mean"].std(ddof=1)
    sd_p = pts["si_etch"].std(ddof=1)
    return dict(tag=tag, w=w, pts=pts, keep=keep, fit=fit,
                lim_all=lim_all, lim_cln=lim_cln, lim_res=lim_res,
                lim_nu=lim_nu, lim_nu_all=lim_nu_all,
                we_raw=we_raw, we_res=we_res, ew=ew,
                sd_wafer=sd_w, sd_point=sd_p,
                sigma_short=lim_cln["sigma"], mean=w["si_etch_mean"].mean(),
                mean_pt=pts["si_etch"].mean())


def report():
    res = {t: analyze(t) for t in ("9pt", "89pt")}
    lim_rows, cap_rows, viol_rows = [], [], []

    print(f"\n{'=' * 78}")
    print("6단계 - 관리도와 공정능력")
    print(f"{'=' * 78}")

    print("\n[1] 특수원인 (표시 후 관리한계 산정에서 제외)")
    for (lot, waf), why in SPECIAL.items():
        print(f"    lot {lot} wafer {waf:2d} : {why}")

    for tag in ("9pt", "89pt"):
        r = res[tag]
        w = r["w"]
        print(f"\n{'-' * 78}\n### {tag}  (웨이퍼 {len(w)}장, lot {w.lot_number.nunique()}개)")

        print("\n[2] 관리한계 - 특수원인 제외 전/후")
        for name, lim in (("제외 전", r["lim_all"]), ("제외 후", r["lim_cln"])):
            print(f"    {name}  CL={lim['center']:.4f}  sigma_MR={lim['sigma']:.4f}"
                  f"  [{lim['lcl']:.4f}, {lim['ucl']:.4f}]  n={lim['n_used']}")
            lim_rows.append({"grid": tag, "chart": "원자료 I", "stage": name, **lim})
        d = 100 * (r["lim_cln"]["sigma"] / r["lim_all"]["sigma"] - 1)
        print(f"    -> 제외로 sigma {d:+.1f}% 변화")

        print("\n[3] 원자료 I-관리도 - 표준 SPC 는 drift 를 감지하는가")
        x = w["si_etch_mean"].to_numpy()
        n_out = int(((x < r["lim_cln"]["lcl"]) | (x > r["lim_cln"]["ucl"])).sum())
        print(f"    3시그마 이탈점 {n_out} / {len(x)}")
        if len(r["we_raw"]):
            print(r["we_raw"]["rule"].value_counts().to_string())
        else:
            print("    WE 런 규칙 위반 없음")
        print(f"    sigma_MR(단기) {r['sigma_short']:.4f} vs sd_overall(장기)"
              f" {r['sd_wafer']:.4f}  = {r['sd_wafer'] / r['sigma_short']:.2f}배")
        print("    -> 장기/단기 비가 1보다 크게 클수록 계통 변동(drift)이 존재")

        print("\n[4] 잔차 I-관리도 (lot + 순번 제거)")
        lim = r["lim_res"]
        print(f"    CL={lim['center']:.4f} sigma={lim['sigma']:.4f}"
              f"  [{lim['lcl']:.4f}, {lim['ucl']:.4f}]   모델 R2={r['fit'].rsquared:.3f}")
        lim_rows.append({"grid": tag, "chart": "잔차 I", "stage": "제외 후", **lim})
        for _, v in w[w["is_special"]].iterrows():
            print(f"    특수원인 lot {int(v.lot_number)} w{int(v.wafer_number)}:"
                  f" 잔차 {v['resid']:+.4f} = {v['resid'] / lim['sigma']:+.2f} sigma")
        if len(r["we_res"]):
            print(r["we_res"]["rule"].value_counts().to_string())

        print("\n[5] EWMA (lam=0.2, lot 마다 리셋) - drift 조기감지")
        fs = first_signal_by_lot(r["ew"])
        print(fs.to_string(index=False))
        det = fs["first_signal_wafer"].dropna()
        print(f"    경보 lot {len(det)}/{len(fs)}"
              + (f", 최초 경보 웨이퍼 순번 중앙값 {det.median():.0f}" if len(det) else ""))

        print("\n[6] 웨이퍼 내 균일도 관리도 (si_nu_pct)")
        lim = r["lim_nu"]
        print(f"    CL={lim['center']:.3f}%  sigma={lim['sigma']:.3f}"
              f"  [{lim['lcl']:.3f}, {lim['ucl']:.3f}]")
        lim_rows.append({"grid": tag, "chart": "균일도 I", "stage": "제외 후", **lim})
        nu = w["si_nu_pct"]
        sd_all, sd_keep = nu.std(), nu[r["keep"]].std()
        print(f"    sd(전체) {sd_all:.3f} vs sd(특수원인 제외) {sd_keep:.3f}"
              f"  -> 전체 sd 는 특수원인 자신이 부풀린 값이라 z 를 과소평가한다")
        hi = w[nu > lim["ucl"]]
        for _, v in hi.iterrows():
            print(f"    이탈 lot {int(v.lot_number)} w{int(v.wafer_number)}:"
                  f" {v.si_nu_pct:.2f}% = {(v.si_nu_pct - lim['center']) / lim['sigma']:+.1f} sigma"
                  f" (sd제외기준 {(v.si_nu_pct - lim['center']) / sd_keep:+.1f})"
                  f"   (평균 잔차로는 {v['resid'] / r['lim_res']['sigma']:+.2f} sigma)")

        for _, v in r["we_raw"].iterrows():
            viol_rows.append({"grid": tag, "chart": "원자료 I", **v.drop("idx").to_dict()})
        for _, v in r["we_res"].iterrows():
            viol_rows.append({"grid": tag, "chart": "잔차 I", **v.drop("idx").to_dict()})

        print("\n[7] 공정능력 - 규격을 발명하지 않고 역산")
        units = {
            "웨이퍼평균/단기": (r["mean"], r["sigma_short"]),
            "웨이퍼평균/장기": (r["mean"], r["sd_wafer"]),
            "전측정점/장기": (r["mean_pt"], r["sd_point"]),
        }
        for tgt in (1.00, 1.33, 1.67, 2.00):
            parts = []
            for lab, (m, s) in units.items():
                half, pct = tolerance_for(s, m, tgt)
                parts.append(f"{lab} +-{half:.2f}um({pct:.2f}%)")
            print(f"    Cpk {tgt:.2f} 필요공차 : " + " | ".join(parts))

        print("\n    참고 - 대표 공차에서의 Cp/Cpk")
        for pct in (3, 5):
            for lab, (m, s) in units.items():
                c = capability(w["si_etch_mean"] if "웨이퍼" in lab else r["pts"]["si_etch"],
                               pct, sigma=s, label=lab)
                cap_rows.append({"grid": tag, **c})
                print(f"      +-{pct}%  {lab:14s} Cp={c['cp']:.2f} Cpk={c['cpk']:.2f}"
                      f"  불량 {c['ppm']:,.0f} ppm")

        sh = stats.shapiro(r["w"]["resid"])
        print(f"\n    정규성(잔차) Shapiro p={sh.pvalue:.4f}"
              f" -> {'기각, Cpk 는 근사로만 해석' if sh.pvalue < 0.05 else '기각 못 함'}")

    print(f"\n{'=' * 78}")
    print("[8] 결론 - 능력을 깎아먹는 주범은 drift 가 아니라 웨이퍼 내 불균일")
    for tag in ("9pt", "89pt"):
        r = res[tag]
        print(f"    {tag:5s} 전측정점 sd {r['sd_point']:.3f} vs 웨이퍼평균 sd"
              f" {r['sd_wafer']:.3f} = {r['sd_point'] / r['sd_wafer']:.1f}배")
    r = res["89pt"]
    c_w = capability(r["w"]["si_etch_mean"], 3, sigma=r["sd_wafer"])
    c_p = capability(r["pts"]["si_etch"], 3, sigma=r["sd_point"])
    print(f"    89pt +-3% 기준 Cp : 웨이퍼평균 {c_w['cp']:.2f} vs 전측정점 {c_p['cp']:.2f}")
    print("    drift 10장 누적 -1.07 um  <<  웨이퍼 내 범위 약 12 um")
    print("    -> 개선 1순위는 반경 프로파일(5단계), 2순위가 drift(4단계)")

    pd.DataFrame(lim_rows).to_csv(PROC / "spc_control_limits.csv", index=False)
    pd.DataFrame(viol_rows).to_csv(PROC / "spc_violations.csv", index=False)
    pd.DataFrame(cap_rows).to_csv(PROC / "spc_capability.csv", index=False)

    specs = {}
    for tag in ("9pt", "89pt"):
        r = res[tag]
        specs[f"{tag}_웨이퍼평균_단기"] = (r["mean"], r["sigma_short"])
        specs[f"{tag}_웨이퍼평균_장기"] = (r["mean"], r["sd_wafer"])
        specs[f"{tag}_전측정점"] = (r["mean_pt"], r["sd_point"])
    tc = tolerance_curve(specs)
    tc.to_csv(PROC / "spc_tolerance_curve.csv", index=False)
    res["curve"] = tc
    res["specs"] = specs
    return res


# ---------------------------------------------------------------- 그림


def _lot_bands(ax, w):
    """lot 경계 표시."""
    for lot, g in w.groupby("lot_number"):
        if lot % 2 == 0:
            ax.axvspan(g["run"].min() - 0.5, g["run"].max() + 0.5, color="grey", alpha=0.07)
    ax.set_xlabel("실행 순번 (lot 순 x 웨이퍼 순)")


def _ichart(ax, w, col, lim, title, ylabel, marker_special=True):
    ax.axhspan(lim["lcl"], lim["ucl"], color="#4C72B0", alpha=0.06)
    ax.axhline(lim["center"], color="k", lw=1.1)
    for y, ls in ((lim["ucl"], "--"), (lim["lcl"], "--")):
        ax.axhline(y, color="#C44E52", lw=1.2, ls=ls)
    ax.plot(w["run"], w[col], "o-", ms=3.4, lw=0.9, color="#4C72B0", zorder=3)
    out = w[(w[col] < lim["lcl"]) | (w[col] > lim["ucl"])]
    ax.scatter(out["run"], out[col], s=70, facecolor="none", edgecolor="#C44E52",
               linewidth=1.8, zorder=5)
    if marker_special:
        sp = w[w["is_special"]]
        ax.scatter(sp["run"], sp[col], marker="s", s=46, facecolor="none",
                   edgecolor="darkorange", linewidth=1.8, zorder=6)
    _lot_bands(ax, w)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def plot_charts(res, tag="89pt"):
    style()
    r = res[tag]
    w = r["w"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)

    _ichart(axes[0], w, "si_etch_mean", r["lim_cln"],
            f"(a) 원자료 I-관리도 - 3시그마 이탈 "
            f"{int(((w.si_etch_mean < r['lim_cln']['lcl']) | (w.si_etch_mean > r['lim_cln']['ucl'])).sum())}건"
            "   표준 관리도는 lot 내 drift 를 관리한계 안에 숨긴다",
            "웨이퍼 평균 Si 식각 (um)")
    for _, g in w.groupby("lot_number"):
        axes[0].plot(g["run"], np.poly1d(np.polyfit(g["run"], g["si_etch_mean"], 1))(g["run"]),
                     color="darkorange", lw=1.3, alpha=0.85, zorder=4)
    axes[0].plot([], [], color="darkorange", lw=1.3,
                 label=f"lot 내 추세 ({r['fit'].params['wafer_number']:+.3f} um/장)")
    axes[0].scatter([], [], marker="s", s=46, facecolor="none", edgecolor="darkorange",
                    linewidth=1.8, label="특수원인 (한계 산정 제외)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", ncol=2)

    _ichart(axes[1], w, "resid", r["lim_res"],
            "(b) 잔차 I-관리도 (lot + 순번 제거) - 계통 성분을 걷어내면 특수원인이 드러난다",
            "잔차 (um)")

    ew = r["ew"]
    ax = axes[2]
    ax.axhline(r["lim_cln"]["center"], color="k", lw=1.1)
    ax.plot(ew["run"], ew["ucl"], color="#C44E52", lw=1.0, ls="--")
    ax.plot(ew["run"], ew["lcl"], color="#C44E52", lw=1.0, ls="--")
    ax.plot(ew["run"], ew["ewma"], "o-", ms=3.4, lw=1.2, color="#55A868", zorder=3)
    sig = ew[ew["signal"]]
    ax.scatter(sig["run"], sig["ewma"], s=70, facecolor="none", edgecolor="#C44E52",
               linewidth=1.8, zorder=5)
    _lot_bands(ax, w)
    ax.set_ylabel("EWMA (um)")
    ax.set_title(f"(c) EWMA lam=0.2, lot 마다 리셋 - 경보 {int(ew['signal'].sum())}건"
                 "   같은 데이터에서 drift 가 보인다", fontsize=10)

    ax = axes[3]
    _ichart(ax, w, "si_nu_pct", r["lim_nu"],
            "(d) 웨이퍼 내 균일도 관리도 - 평균 차트가 놓치는 이상을 잡는다",
            "비균일도 (max-min)/mean (%)")
    lim_nu = r["lim_nu"]
    # 최대 이상점 하나가 축을 독점해 나머지 구조가 안 보이므로 y 축을 잘라내고
    # 축 밖으로 나간 점은 상단에 삼각 마커 + 실제 값으로 표기한다.
    body = w.loc[~w["is_special"], "si_nu_pct"]
    lo = min(body.min(), lim_nu["lcl"])
    hi_y = max(body.max(), lim_nu["ucl"])
    pad = (hi_y - lo) * 0.35
    ax.set_ylim(lo - pad, hi_y + pad)
    top = ax.get_ylim()[1]
    over = w[w["si_nu_pct"] > top]
    ax.scatter(over["run"], [top] * len(over), marker="^", s=90, color="#C44E52", zorder=7,
               clip_on=False)
    for _, v in over.iterrows():
        ax.annotate(f"lot {int(v.lot_number)} w{int(v.wafer_number)}  {v.si_nu_pct:.1f}%\n"
                    f"(축 밖, {(v.si_nu_pct - lim_nu['center']) / lim_nu['sigma']:+.0f} sigma)",
                    (v["run"], top), fontsize=8.5, ha="left", va="top", color="#C44E52",
                    fontweight="bold", xytext=(10, -4), textcoords="offset points")
    out = w[(w["si_nu_pct"] > lim_nu["ucl"]) & (w["si_nu_pct"] <= top)]
    for i, (_, v) in enumerate(out.iterrows()):
        ax.annotate(f"lot {int(v.lot_number)} w{int(v.wafer_number)}\n"
                    f"{(v.si_nu_pct - lim_nu['center']) / lim_nu['sigma']:+.1f} sigma",
                    (v["run"], v["si_nu_pct"]), fontsize=8, ha="center",
                    xytext=(0, -30 - 20 * (i % 2)), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    fig.suptitle(f"관리도 3종 + 균일도 ({tag}) - 표준 SPC 는 chamber drift 를 감지하지 못한다",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "spc_control_charts.png", dpi=150)
    plt.close(fig)


def plot_capability(res):
    style()
    tc, specs = res["curve"], res["specs"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    ax = axes[0]
    styles = {
        "89pt_웨이퍼평균_단기": ("#4C72B0", "-", "89점 웨이퍼평균 / 단기(sigma_MR)"),
        "89pt_웨이퍼평균_장기": ("#4C72B0", "--", "89점 웨이퍼평균 / 장기(전체 sd)"),
        "89pt_전측정점": ("#C44E52", "-", "89점 전측정점 (소자 규격 관점)"),
        "9pt_웨이퍼평균_장기": ("#55A868", "--", "9점 웨이퍼평균 / 장기"),
        "9pt_전측정점": ("#8172B2", "-", "9점 전측정점"),
    }
    for k, (c, ls, lab) in styles.items():
        ax.plot(tc["tol_pct"], tc[k], color=c, ls=ls, lw=1.8, label=lab)
    for y, lab in ((1.33, "Cpk 1.33 (통상 요구)"), (1.0, "Cpk 1.00")):
        ax.axhline(y, color="grey", lw=0.9, ls=":")
        ax.text(7.9, y + 0.05, lab, fontsize=7.5, ha="right", color="dimgrey")
    ax.set_xlim(0.5, 8)
    ax.set_ylim(0, 4)
    ax.set_xlabel("규격 공차 반폭 (평균 대비 %)")
    ax.set_ylabel("Cpk")
    ax.set_title("(a) 공차 폭 vs Cpk - 규격이 없으므로 역산으로 답한다", fontsize=10)
    # 각 곡선이 Cpk 1.33 을 넘는 공차를 축 위에 직접 표시한다
    for k, (c, _, _) in styles.items():
        x = np.interp(1.33, tc[k], tc["tol_pct"], left=np.nan, right=np.nan)
        if np.isfinite(x) and x <= 8:
            ax.plot([x, x], [0, 1.33], color=c, lw=0.8, ls=":", alpha=0.7)
            ax.annotate(f"{x:.1f}%", (x, 0), color=c, fontsize=8, fontweight="bold",
                        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    ax = axes[1]
    labels, cps, colors = [], [], []
    for tag, col in (("9pt", "#55A868"), ("89pt", "#4C72B0")):
        r = res[tag]
        for lab, s, m in (("웨이퍼평균\n(공정관리 관점)", r["sd_wafer"], r["mean"]),
                          ("전측정점\n(소자규격 관점)", r["sd_point"], r["mean_pt"])):
            labels.append(f"{tag}\n{lab}")
            cps.append((m * 3 / 100) / (3 * s))
            colors.append(col if "웨이퍼" in lab else "#C44E52")
    b = ax.bar(range(len(cps)), cps, color=colors, width=0.62)
    ax.bar_label(b, fmt="%.2f", fontsize=9.5, padding=2)
    ax.axhline(1.33, color="grey", ls=":", lw=1.0)
    ax.text(len(cps) - 0.4, 1.38, "Cpk 1.33", fontsize=8, ha="right", color="dimgrey")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Cp (공차 +-3% 기준)")
    ax.set_title("(b) 같은 공정, 같은 공차 - 무엇을 측정 단위로 보느냐로 9배 차이", fontsize=10)
    ax.text(0.02, 0.96, "웨이퍼 평균만 보면 능력 있어 보이지만\n"
                        "소자는 웨이퍼 위 모든 점에서 규격을 만족해야 한다",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(fc="white", alpha=0.85, ec="lightgrey"))

    fig.suptitle("공정능력 - 능력을 깎아먹는 주범은 drift 가 아니라 웨이퍼 내 반경 프로파일",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "spc_capability.png", dpi=150)
    plt.close(fig)


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = report()
    plot_charts(res, "89pt")
    plot_capability(res)
    print(f"\n저장: {FIG / 'spc_control_charts.png'}")
    print(f"저장: {FIG / 'spc_capability.png'}")


if __name__ == "__main__":
    main()
