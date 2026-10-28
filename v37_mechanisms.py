# -*- coding: utf-8 -*-
"""v37 — 봇 메커니즘 4종(문헌·블로그 조사): 국면 필터 / 변동성 맞춤 비중 / 위험 조정 순위 / 미너비니 템플릿.

━━━ 후보 (파라미터는 문헌값 고정, 이 데이터로 고르지 않음) ━━━
  REG   SPY 종가 > SPY 200일 이동평균일 때만 신규 매수(보유 종목은 샹들리에대로). 클레나우 규칙.
  VOL   변동성 맞춤 비중: 금액 = min(15%, 0.5% × 가격 ÷ (k·ATR14)) × 자본. k = 그 창의 샹들리에 배수.
        손절 시 자본의 약 0.5% 손실이 되게 맞춘다(터틀 '유닛'). 슬롯 수 10 은 그대로, 남는 돈은 현금.
  RADJ  순위 = K일 수익률 ÷ 같은 K일 일간수익률 표준편차 (Rachev 외 2007).
  MIN   미너비니 추세 템플릿을 진입 조건에 추가: 종가 > SMA150·SMA200, SMA150 > SMA200, SMA50 > 둘 다,
        SMA200 > 22일 전 SMA200, 종가 ≥ 1.30×52주 저가, 종가 ≥ 0.75×52주 고가,
        RS(0.4·R63+0.2·R126+0.2·R189+0.2·R252 의 단면 백분위) ≥ 70%.
각 변형은 그 규칙을 켠 채 학습 창 파라미터를 다시 고른다(우리 봇과 같은 분기 재선택).

━━━ 구간 ━━━  A 2016-05~2021-03 (v35 주가) / B 2022-09~2024-12 (v28 검증 창). 봉인 구간 미사용.

━━━ 사전 판정 기준 (결과 보기 전, 2026-10-29 커밋으로 고정) ━━━
  A·B 둘 다에서 Sharpe ≥ 우리 봇 + 0.20, MDD 가 5%p 넘게 깊지 않음 → '실전 후보'(사용자 승인 후 페이퍼 먼저).
  그 외 기각. 조합(예: REG+VOL)은 이번에 보지 않는다(다중 검정 줄이기).
  ⚠ 생존편향 유니버스. 무작위 거부만으로 SR ±0.4 출렁인 적 있음(v34b).

사용: python3 v37_mechanisms.py
"""
import itertools
import pickle

import numpy as np
import pandas as pd

import v28_walkforward as v28
import v35_8k_replicate as v35
from v26_us_holdout import build as build_b
from v36_exit_rules import with_low

SLOTS = 10
MARGIN, MDD_TOL = 0.20, 5.0
VARIANTS = ["REG", "VOL", "RADJ", "MIN"]


def extras(ind):
    cl, hi, lo = ind["cl"], ind["hi"], ind["lo"]
    spy = pickle.load(open("cache/v37_spy.pkl", "rb"))["Close"].reindex(cl.index).ffill()
    ind["reg"] = (spy > spy.rolling(200).mean()).values
    dr = cl.pct_change(fill_method=None)
    ind["vol"] = {k: dr.rolling(k).std() for k in v28.GRID["mom"]}
    s50, s150, s200 = (cl.rolling(n).mean() for n in (50, 150, 200))
    h52, l52 = hi.rolling(252).max(), lo.rolling(252).min()
    raw = 0.4 * cl.pct_change(63, fill_method=None) + 0.2 * cl.pct_change(126, fill_method=None) \
        + 0.2 * cl.pct_change(189, fill_method=None) + 0.2 * cl.pct_change(252, fill_method=None)
    rs = raw.rank(axis=1, pct=True)
    tmpl = (cl > s150) & (cl > s200) & (s150 > s200) & (s50 > s150) & (s50 > s200) \
        & (s200 > s200.shift(22)) & (cl >= 1.30 * l52) & (cl >= 0.75 * h52) & (rs >= 0.70)
    ind["minv"] = tmpl.fillna(False).values
    return ind


def sim(ind, cfg, slots, i0, i1, var=None, cap=100.0):
    O, H, C = ind["op"].values, ind["hi"].values, ind["cl"].values
    A = ind["atr"].values
    D = ind["dc"][cfg["dc"]].values
    M = ind["ma"][cfg["ma"]].values
    R = ind["mom"][cfg["mom"]].values
    if var == "RADJ":
        sd = ind["vol"][cfg["mom"]].values
        with np.errstate(invalid="ignore", divide="ignore"):
            R = R / sd
    slot = cap / slots
    cash, pos, eq, tr = cap, {}, [], []
    for i in range(i0, min(i1, len(C) - 1)):
        for j in list(pos):
            p = pos[j]
            if np.isnan(H[i, j]) or np.isnan(A[i, j]):
                continue
            p["pk"] = max(p["pk"], H[i, j])
            if C[i, j] <= p["pk"] - cfg["chand"] * A[i, j]:
                px = O[i + 1, j]
                if not (px > 0):
                    continue
                cash += p["q"] * px * (1 - v28.COST / 100)
                tr.append((px / p["e"] - 1) * 100 - v28.COST)
                del pos[j]
        if len(pos) < slots and not (var == "REG" and not ind["reg"][i]):
            nx = O[i + 1]
            ok = (C[i] > D[i]) & (C[i] > M[i]) & ~np.isnan(R[i]) & ~np.isnan(A[i]) & (nx > 0)
            if var == "MIN":
                ok &= ind["minv"][i]
            for j in np.argsort(-np.where(ok, R[i], -np.inf)):
                if len(pos) >= slots or not ok[j]:
                    break
                if j in pos:
                    continue
                amt = slot
                if var == "VOL":
                    amt = min(0.15 * cap, 0.005 * cap * C[i, j] / (cfg["chand"] * A[i, j]))
                if cash < amt * 0.99 or amt <= 0:
                    break
                cash -= amt
                pos[j] = {"q": amt / nx[j], "e": nx[j], "pk": nx[j]}
        eq.append(cash + sum(p["q"] * C[i, j] for j, p in pos.items() if not np.isnan(C[i, j])))
    return pd.Series(eq).pct_change().dropna().values, tr


def walkforward(ind, combos, idx, start, end, var=None):
    s0, s1 = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end))
    rets, trades, a = [], [], s0
    while a + v28.TRAIN + v28.TEST <= s1:
        best, bsr = None, -9e9
        for cfg in combos:
            sr = v28.sharpe(sim(ind, cfg, SLOTS, a, a + v28.TRAIN, var)[0])
            if sr > bsr:
                best, bsr = cfg, sr
        r, t = sim(ind, best, SLOTS, a + v28.TRAIN, a + v28.TRAIN + v28.TEST, var)
        rets.append(r)
        trades += t
        a += v28.TEST
    r = np.concatenate(rets)
    eq = (1 + pd.Series(r)).cumprod()
    return {"tot": (eq.iloc[-1] - 1) * 100, "sr": v28.sharpe(r),
            "mdd": float((eq / eq.cummax() - 1).min()) * 100, "n": len(trades),
            "win": 100 * float(np.mean(np.array(trades) > 0)) if trades else 0.0}


def main():
    v28.COST = 0.20
    combos = [dict(zip(v28.GRID, v)) for v in itertools.product(*v28.GRID.values())]
    windows = {
        "A 2016~2021": (extras(with_low(v35.build(), v35.PX)), v35.START, v35.END),
        "B 2022~2024": (extras(with_low(build_b(), "cache/v26_us_px.pkl")), v28.WF_START, v28.WF_END),
    }
    res = {}
    for w, (ind, st, en) in windows.items():
        idx = ind["cl"].index
        cfg = {"dc": 40, "ma": 120, "chand": 3.0, "mom": 20}
        s0 = idx.searchsorted(pd.Timestamp(st)) + 260
        r1, t1 = sim(ind, cfg, SLOTS, s0, s0 + 300)
        r2, t2 = v28.sim(ind, cfg, SLOTS, s0, s0 + 300)
        assert len(t1) == len(t2) and np.allclose(r1, r2), "시뮬레이터 불일치"
        res[w] = {"우리 봇": walkforward(ind, combos, idx, st, en)}
        for v in VARIANTS:
            res[w][v] = walkforward(ind, combos, idx, st, en, v)
        print(f"\n[v37] {w} (정합성 OK, 국면 ON 비율 {np.mean(ind['reg'][idx.searchsorted(pd.Timestamp(st)):]) * 100:.0f}%)")
        print(f"   {'':<8}{'누적%':>8}{'Sharpe':>8}{'MDD%':>8}{'거래':>6}{'승률%':>7}")
        for k, s in res[w].items():
            print(f"   {k:<8}{s['tot']:>8.1f}{s['sr']:>8.2f}{s['mdd']:>8.1f}{s['n']:>6}{s['win']:>7.0f}")
    print("\n  ── 사전 기준 (A·B 둘 다 SR +0.20 이상, MDD 5%p 이내) ──")
    for v in VARIANTS:
        ok = all(res[w][v]["sr"] >= res[w]["우리 봇"]["sr"] + MARGIN
                 and res[w][v]["mdd"] >= res[w]["우리 봇"]["mdd"] - MDD_TOL for w in res)
        det = " / ".join(f"{w[:1]} {res[w][v]['sr']:.2f}({res[w][v]['mdd']:.0f}%) vs "
                         f"{res[w]['우리 봇']['sr']:.2f}({res[w]['우리 봇']['mdd']:.0f}%)" for w in res)
        print(f"   {v:<6}{'실전 후보' if ok else '기각':<8}{det}")


if __name__ == "__main__":
    main()
