# -*- coding: utf-8 -*-
"""v32 — 외부(문헌·GitHub) 추세·모멘텀 방법을 우리 미장 봇과 비교.

━━━ 후보 (전부 저자가 정한 고정 파라미터, 우리가 이 데이터로 고르지 않음) ━━━
 C1 클레나우 "Stocks on the Move"(2015): 90일 로그가격 회귀 기울기(연율) × R² 순위.
    거르기 = 종가 > SMA100, 90일 내 하루 ±15% 이상 움직임 없음. 국면 = 지수 > 200일선일 때만 매수.
    매주 수요일 교체: 상위 20% 밖 · SMA100 아래 · 갭이면 매도.
    ⚠ 원본은 ATR 위험균등으로 수십 종목 → $77 계좌라 **10종목 균등**으로 옮김.
 C2 52주 신고가 근접도(George & Hwang 2004): 우리 봇의 순위 기준만 '종가 ÷ 52주 고가'로
    바꾸고 나머지(돌파·이평·샹들리에·분기 재선택)는 그대로.
 C3 퀀트 모멘텀(Gray & Vogel 2016): 12-2개월 수익률 상위 10% → 정보 이산도(ID, Da 외 2014)
    낮은 절반 → 그중 ID 가장 낮은 10종목. 2·5·8·11월 마지막 거래일에 교체.
    ⚠ 원본은 약 50종목 → 10종목.
 C4 변동성 맞춤(Barroso & Santa-Clara 2015)을 우리 봇에 덧씌움: 비중 = min(1, 12% ÷ 직전
    126일 실현 변동성). 나머지는 현금(수익 0).
 C5 손절폭 10×ATR 고정(Wilcox & Crittenden 2005): 우리 봇에서 샹들리에 배수만 10 으로 고정,
    나머지 파라미터는 분기 재선택 그대로.
 C6 ETF 기준선(미장만): GEM 듀얼 모멘텀(Antonacci), VAA-G4(Keller & Keuning). 월말 교체,
    교체 시 편도 0.1%.

━━━ 구간 ━━━
v28 워크포워드 검증 창과 같은 날짜(미 2022-09~2024-12, 국 2022-09~2024-09).
봉인 구간(2025~)은 쓰지 않는다 — 미장은 세 번 열었고 다시 열지 않는다. ETF 가격도 2024 까지만 받았다.

━━━ 사전 판정 기준 (결과 보기 전, 2026-10-11 커밋으로 고정) ━━━
비교 상대 = 우리 봇(슬롯 10, 분기 재선택 워크포워드).
 · 교체 후보: 미장·국장 **둘 다**에서 Sharpe ≥ 우리 봇 + 0.20, MDD 가 우리 봇보다 5%p 넘게
   깊지 않음, Sharpe > 시장(유니버스 동일가중).
 · 페이퍼 관찰: 둘 다에서 Sharpe ≥ 우리 봇, Sharpe > 시장.
 · 그 외 기각.
 · ETF(C6)는 미장 하나뿐이므로 최대 '페이퍼 관찰'.
'교체 후보'라도 자동 교체하지 않는다 — 사용자에게 제안하고, 먼저 페이퍼로 돌린다.
⚠ 2년 남짓 상승장 구간이라 Sharpe 0.3 미만 차이는 통계적으로 구분이 안 된다.
⚠ 유니버스가 현재 구성종목이라 생존편향이 모든 주식 전략에 같이 끼어 있다.

━━━ 결과 (2026-10-11, 기준 커밋 38a47a6 이후 실행) — 전부 기각, 우리 봇 유지 ━━━
                      미 SR(누적/MDD)        국 SR(누적/MDD)
    우리 봇            1.43 (+99.7/−15.1)    0.49 (+23.2/−31.2)
    시장               1.41 (+61.5/−13.2)    0.78 (+30.6/−16.4)
    C1 클레나우        1.65 (+105.2/−15.2)   0.56 (+23.1/−21.7)  ← 둘 다 우리 봇 이상이나 국장 시장 미달
    C2 52주고가 순위   1.42 (+60.0/−13.0)    0.79 (+43.6/−22.9)  ← 미장 0.01 차(잡음 수준)
    C3 퀀트 모멘텀     1.33 (+73.2/−15.6)    1.21 (+94.2/−22.2)
    C4 +변동성 맞춤    1.25 (+40.4/−11.8)    0.06 (−0.3/−19.7)   평균비중 0.54 / 0.41
    C5 +10×ATR         0.95 (+57.2/−17.1)    0.28 (+7.7/−27.6)
    C6 GEM             1.22 (+33.3/−11.8)    —
    C6 VAA-G4          0.27 (+4.7/−9.0)      —
두 시장 모두에서 우리 봇을 뚜렷이(+0.2) 이기는 방법은 없었다. 차이 대부분이 잡음 범위다.

사용: python3 v32_external.py
"""
import itertools
import pickle

import numpy as np
import pandas as pd

import v28_walkforward as v28
import v30_kis_strategies as v30
from v22_htf_kr import _kospi
from v26_us_holdout import build

SLOTS = 10
MARGIN, MDD_TOL = 0.20, 5.0


def window(idx):
    s0 = idx.searchsorted(pd.Timestamp(v28.WF_START))
    s1 = idx.searchsorted(pd.Timestamp(v28.WF_END))
    a0, a = s0 + v28.TRAIN, s0
    while a + v28.TRAIN + v28.TEST <= s1:
        a += v28.TEST
    return a0, a + v28.TRAIN


def wf_series(ind, combos, idx):
    """v28.walkforward 와 같은 절차인데 일별 수익률을 이어 붙여 돌려준다."""
    s0 = idx.searchsorted(pd.Timestamp(v28.WF_START))
    s1 = idx.searchsorted(pd.Timestamp(v28.WF_END))
    rets, trades, a = [], [], s0
    while a + v28.TRAIN + v28.TEST <= s1:
        best, best_sr = None, -9e9
        for cfg in combos:
            r, _ = v28.sim(ind, cfg, SLOTS, a, a + v28.TRAIN)
            sr = v28.sharpe(r)
            if sr > best_sr:
                best, best_sr = cfg, sr
        r, tr = v28.sim(ind, best, SLOTS, a + v28.TRAIN, a + v28.TRAIN + v28.TEST)
        rets.append(r)
        trades += tr
        a += v28.TEST
    return np.concatenate(rets), trades


def regression_score(cl, n=90):
    """클레나우 점수 = (exp(기울기)^252 − 1) × R², 로그가격을 0..n−1 에 회귀."""
    y = np.log(cl)
    k = pd.DataFrame(np.arange(len(y))[:, None].repeat(y.shape[1], 1),
                     index=y.index, columns=y.columns).astype(float)
    sy = y.rolling(n).sum()
    sy2 = (y * y).rolling(n).sum()
    sky = (k * y).rolling(n).sum()
    start = k - (n - 1)
    sty = sky - start * sy                       # Σ t'·y, t' = 0..n−1
    st, st2 = n * (n - 1) / 2, (n - 1) * n * (2 * n - 1) / 6
    sxx = n * st2 - st ** 2
    sxy = n * sty - st * sy
    syy = n * sy2 - sy ** 2
    slope = sxy / sxx
    r2 = (sxy ** 2) / (sxx * syy)
    return (np.exp(slope) ** 252 - 1) * r2


def clenow(ind, regime):
    cl = ind["cl"]
    score = regression_score(cl)
    sma100 = cl.rolling(100).mean()
    gap = cl.pct_change(fill_method=None).abs().rolling(90).max() >= 0.15
    ok = (cl > sma100) & ~gap & score.notna()
    rank_pct = score.where(ok).rank(axis=1, pct=True, ascending=False)
    top = rank_pct <= 0.20
    reb = pd.Series(cl.index.weekday == 2, index=cl.index)          # 수요일
    rb = reb.values[:, None]
    reg = regime.reindex(cl.index).ffill()
    reg_on = (reg > reg.rolling(200).mean()).values[:, None]
    buy = rb & reg_on & (ok & top).fillna(False).values
    sell = rb & ~(ok & top).fillna(False).values
    return buy, score.values, sell


def quant_momentum(ind):
    cl = ind["cl"]
    m = cl.shift(21) / cl.shift(252) - 1
    dr = cl.pct_change(fill_method=None)
    pos = (dr > 0).astype(float).shift(21).rolling(231).mean()
    neg = (dr < 0).astype(float).shift(21).rolling(231).mean()
    idisc = np.sign(m) * (neg - pos)
    valid = m.notna() & idisc.notna()
    dec = m.where(valid).rank(axis=1, pct=True, ascending=False) <= 0.10
    idr = idisc.where(dec).rank(axis=1, ascending=True)             # 낮을수록 좋음
    half = idisc.where(dec).rank(axis=1, pct=True, ascending=True) <= 0.5
    target = half & (idr <= SLOTS)
    d = cl.index
    last = pd.Series(d, index=d).groupby([d.year, d.month]).transform("max") == d
    reb = (last & d.month.isin([2, 5, 8, 11])).values[:, None]
    buy = reb & target.fillna(False).values
    sell = reb & ~target.fillna(False).values
    return buy, (-idisc).values, sell


def vol_scale(r, target=0.12, n=126):
    s = pd.Series(r)
    vol = s.rolling(n, min_periods=20).std().shift(1) * np.sqrt(252)
    w = (target / vol).clip(upper=1.0).fillna(1.0)
    return (s * w).values, float(w.mean())


def etf_strats(idx_dates):
    px = pickle.load(open("cache/v32_etf.pkl", "rb"))
    P = pd.DataFrame(px).sort_index().ffill()
    dr = P.pct_change(fill_method=None).fillna(0)
    me = P.resample("ME").last()
    r = lambda k: me / me.shift(k) - 1
    out = {}
    # GEM
    hold = []
    for t in me.index:
        if any(pd.isna(r(12).loc[t, s]) for s in ("SPY", "EFA", "BIL")):
            hold.append(None)
            continue
        if r(12).loc[t, "SPY"] > r(12).loc[t, "BIL"]:
            hold.append("SPY" if r(12).loc[t, "SPY"] >= r(12).loc[t, "EFA"] else "EFA")
        else:
            hold.append("AGG")
    out["GEM 듀얼 모멘텀"] = pd.Series(hold, index=me.index)
    # VAA-G4
    sc = 12 * r(1) + 4 * r(3) + 2 * r(6) + r(12)
    hold = []
    for t in me.index:
        row = sc.loc[t]
        if row[["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY"]].isna().any():
            hold.append(None)
            continue
        off = row[["SPY", "EFA", "EEM", "AGG"]]
        hold.append(off.idxmax() if (off > 0).all() else row[["LQD", "IEF", "SHY"]].idxmax())
    out["VAA-G4"] = pd.Series(hold, index=me.index)

    res = {}
    for name, h in out.items():
        daily = h.reindex(P.index, method="ffill").shift(1)          # 월말 결정 → 다음 날부터
        rr = pd.Series(0.0, index=P.index)
        prev = None
        for t in P.index:
            cur = daily.loc[t]
            if isinstance(cur, str):
                rr.loc[t] = dr.loc[t, cur] - (0.002 if prev not in (None, cur) else 0.0)
            prev = cur if isinstance(cur, str) else prev
        rr = rr[(rr.index >= idx_dates[0]) & (rr.index <= idx_dates[1])]
        res[name] = v30.stats(rr.values, [])
    return res


def evaluate(market):
    if market == "kr":
        v30.COST = v28.COST = 0.23
        ind = v30.build_kr()
        regime = _kospi()
    else:
        v30.COST = v28.COST = 0.20
        ind = build()
        regime = pd.Series(pickle.load(open("cache/v32_etf.pkl", "rb"))["SPY"])
    idx = ind["cl"].index
    a0, a1 = window(idx)
    combos = [dict(zip(v28.GRID, v)) for v in itertools.product(*v28.GRID.values())]
    out = {}

    r, tr = wf_series(ind, combos, idx)
    out["우리 봇(분기 재선택)"] = v30.stats(r, tr)
    out["시장(동일가중)"] = v30.stats(ind["bench"].iloc[a0:a1].pct_change().dropna().values, [])

    b, s, sl = clenow(ind, regime)
    out["C1 클레나우"] = v30.stats(*v30.run(ind, b, s, sl, a0, a1))

    hi = ind["hi"]
    ind["mom"]["h52"] = ind["cl"] / hi.rolling(252, min_periods=126).max()
    c2 = [c | {"mom": "h52"} for c in combos if c["mom"] == v28.GRID["mom"][0]]
    out["C2 52주고가 순위"] = v30.stats(*wf_series(ind, c2, idx))

    b, s, sl = quant_momentum(ind)
    out["C3 퀀트 모멘텀"] = v30.stats(*v30.run(ind, b, s, sl, a0, a1))

    rs, wbar = vol_scale(r)
    out["C4 우리 봇+변동성 맞춤"] = v30.stats(rs, tr) | {"w": wbar}

    c5 = [c | {"chand": 10.0} for c in combos if c["chand"] == v28.GRID["chand"][0]]
    out["C5 우리 봇+10×ATR"] = v30.stats(*wf_series(ind, c5, idx))

    if market == "us":
        out.update({f"C6 {k}": v for k, v in etf_strats((idx[a0], idx[a1 - 1])).items()})
    return out, (idx[a0].date(), idx[a1 - 1].date())


def main():
    res = {}
    for m in ("us", "kr"):
        res[m], span = evaluate(m)
        print(f"\n[v32] {'미장' if m == 'us' else '국장'} {span[0]} ~ {span[1]}")
        print(f"  {'':<34}{'누적%':>8}{'Sharpe':>8}{'MDD%':>8}{'거래':>6}")
        print("  " + "-" * 64)
        for k, s in res[m].items():
            w = f"  평균비중 {s['w']:.2f}" if "w" in s else ""
            print(f"  {k:<34}{s['tot']:>8.1f}{s['sr']:>8.2f}{s['mdd']:>8.1f}{s['n']:>6}{w}")

    print("\n  ── 사전 기준 판정 ──")
    base = {m: res[m]["우리 봇(분기 재선택)"] for m in res}
    mkt = {m: res[m]["시장(동일가중)"] for m in res}
    for k in res["us"]:
        if k.startswith(("우리", "시장")):
            continue
        ms = [m for m in ("us", "kr") if k in res[m]]
        s = {m: res[m][k] for m in ms}
        strong = all(s[m]["sr"] >= base[m]["sr"] + MARGIN and s[m]["mdd"] >= base[m]["mdd"] - MDD_TOL
                     and s[m]["sr"] > mkt[m]["sr"] for m in ms)
        weak = all(s[m]["sr"] >= base[m]["sr"] and s[m]["sr"] > mkt[m]["sr"] for m in ms)
        if strong and len(ms) == 2:
            v = "교체 후보"
        elif weak:
            v = "페이퍼 관찰"
        else:
            v = "기각"
        detail = " / ".join(f"{'미' if m == 'us' else '국'} SR {s[m]['sr']:.2f} vs {base[m]['sr']:.2f}"
                            for m in ms)
        print(f"  {k[:30]:<32}{v:<10}{detail}")


if __name__ == "__main__":
    main()
