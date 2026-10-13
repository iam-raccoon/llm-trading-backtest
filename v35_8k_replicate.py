# -*- coding: utf-8 -*-
"""v35 — v34 '악재 8-K 뒤 신규 매수 거부(V1)'가 **다른 5년(2016-06 ~ 2021-05)에서도 재현되나.**

━━━ 왜 ━━━
v34(2022-09~2024-12)에서 V1 이 SR 1.43 → 1.94, 무작위 거부 20회 전부보다 높았다. 다만 사전 기준
(사건 분석 예측력)은 불통과였고 결과를 본 뒤의 관찰이라, 실전에 넣기 전에 **한 번도 안 쓴 기간**에서
재현을 본다. 앞으로의 데이터로만 기다리면 2~4년이 걸린다.

━━━ 같게 둔 것 ━━━
사전(LM), 점수식, 200단어 이상, 시점 연결(첫 거래 가능 시가), 악재 = 직전 252거래일 하위 20%,
매수 거부 창 5거래일, 우리 봇(슬롯 10, v28 격자, 학습 252 / 검증 63 워크포워드), 비용 0.20%.
다른 점: 기준값(하위 20%)은 **무작위 30% 표본 공시**로만 추정한다(후보 창 공시만 받으면 분포가 치우치므로).

━━━ 사전 판정 기준 (결과 보기 전, 2026-10-13 커밋으로 고정) ━━━
 R1 사건: 후보 창 공시 중 악재 − 나머지의 20일 시장 대비 초과수익 차이 < 0, Welch t ≤ −1.65
    (같은 방향 재현을 보는 한쪽 검정)
 R2 봇:   V1 Sharpe > 우리 봇 Sharpe 그리고 V1 Sharpe ≥ 무작위 거부 20회 Sharpe 의 80번째 백분위
    (무작위 거부 = 후보 창 공시에서 악재 수와 같은 수를 무작위로 골라 같은 규칙 적용)
 판정: R1·R2 둘 다 → '실전 도입 제안'(사용자 승인 후 반영)
       R2 만     → '약한 재현' — 실전 미반영, 그림자 기록 유지
       그 외     → '재현 실패' — 아이디어 폐기
 ⚠ 유니버스가 지금 구성종목이라 2016 시점 생존편향이 크다(우리 봇·V1·무작위 모두 같은 조건).

사용: python3 v35_8k_replicate.py
"""
import itertools
import json
import os
import pickle

import numpy as np
import pandas as pd

import v28_walkforward as v28
import v34_8k_tone as v34

HERE = os.path.dirname(os.path.abspath(__file__))
PX = os.path.join(HERE, "cache", "v35_us_px.pkl")
SCORES = "/home/user/llm-trading-backtest/cache/v35_8k/scores.jsonl"
START, END = "2015-06-01", "2021-05-31"
SEEDS = 20


def build():
    raw = pickle.load(open(PX, "rb"))
    codes = list(raw)
    f = lambda k: pd.DataFrame({c: raw[c]["df"][k] for c in codes}).sort_index()
    op, hi, lo, cl = f("Open"), f("High"), f("Low"), f("Close")
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo).stack(), (hi - pc).abs().stack(),
                    (lo - pc).abs().stack()], axis=1).max(axis=1).unstack()
    ind = {"op": op, "hi": hi, "cl": cl, "atr": tr.rolling(14).mean(), "codes": codes}
    ind["dc"] = {d: hi.rolling(d).max().shift(1) for d in v28.GRID["dc"]}
    ind["ma"] = {m: cl.rolling(m).mean() for m in v28.GRID["ma"]}
    ind["mom"] = {m: cl / cl.shift(m) - 1 for m in v28.GRID["mom"]}
    ind["bench"] = cl.pct_change(fill_method=None).mean(axis=1).add(1).cumprod()
    return ind


def load_events(idx, codes):
    ev = pd.DataFrame(json.loads(x) for x in open(SCORES))
    ev = ev[(ev["words"] >= v34.MIN_WORDS) & ev["ticker"].isin(codes)].copy()
    ev["tone"] = (ev["pos"] - ev["neg"]) / ev["words"]
    acc = pd.to_datetime(ev["accepted"], utc=True, errors="coerce")
    acc = acc.fillna(pd.to_datetime(ev["date"]).dt.tz_localize("UTC") + pd.Timedelta(hours=21))
    et = acc.dt.tz_convert("America/New_York")
    day = et.dt.tz_localize(None).dt.normalize()
    before = (et.dt.hour * 60 + et.dt.minute) < 9 * 60 + 30
    ps = idx.searchsorted(day.values, side="left")
    pn = idx.searchsorted(day.values, side="right")
    ev["open_pos"] = np.where(before.values, ps, pn)
    ev = ev[(ev["open_pos"] >= 1) & (ev["open_pos"] < len(idx))].copy()
    ev["dec"] = ev["open_pos"] - 1
    ev["col"] = ev["ticker"].map({c: k for k, c in enumerate(codes)})
    ev = ev.sort_values("open_pos").reset_index(drop=True)
    sp = ev[ev["sampled"]]
    sp_pos, sp_tone = sp["open_pos"].values, sp["tone"].values
    thr = np.full(len(ev), np.nan)
    for k, p in enumerate(ev["open_pos"].values):
        m = (sp_pos < p) & (sp_pos >= p - v34.LOOKBACK)
        if m.sum() >= 200:
            thr[k] = np.quantile(sp_tone[m], v34.BAD_Q)
    ev["has_thr"] = ~np.isnan(thr)
    ev["bad"] = ev["tone"].values <= thr
    return ev


def veto_mask(rows, T, N):
    bd = np.zeros((T, N), bool)
    bd[rows["dec"].values, rows["col"].values] = True
    vt = np.zeros((T, N), bool)
    for k in range(v34.VETO_DAYS):
        vt[k:] |= bd[:T - k]
    return vt


def main():
    v28.WF_START, v28.WF_END = START, END
    v28.COST = 0.20
    ind = build()
    idx, codes = ind["cl"].index, ind["codes"]
    ev = load_events(idx, codes)
    cw = ev[ev["cand_window"] & ev["has_thr"]].copy()
    print(f"[v35] 공시 {len(ev)}건 (후보 창 {int(ev['cand_window'].sum())}, 표본 {int(ev['sampled'].sum())})"
          f" | 판정 가능 후보 창 {len(cw)}건 중 악재 {int(cw['bad'].sum())}건")

    # R1 — 후보 창 공시의 사건 분석
    es = v34.event_study(cw, ind)
    s = es[20]
    r1 = s["diff"] < 0 and s["t"] <= -1.65
    print("\n  ── R1 사건 분석 (후보 창 공시, 시장 대비 초과수익 %) ──")
    for h, x in es.items():
        print(f"   +{h:>2}일  악재 {x['bad']:+.2f} (n={x['bad_n']})  나머지 {x['rest']:+.2f} "
              f"(n={x['rest_n']})  차이 {x['diff']:+.2f}  t={x['t']:+.2f}")

    # R2 — 봇
    T, N = ind["cl"].shape
    combos = [dict(zip(v28.GRID, v)) for v in itertools.product(*v28.GRID.values())]
    base = v34.walkforward(ind, combos, idx)
    v1 = v34.walkforward(ind, combos, idx, veto=veto_mask(cw[cw["bad"]], T, N))
    nbad = int(cw["bad"].sum())
    plac = []
    for sd in range(SEEDS):
        plac.append(v34.walkforward(ind, combos, idx,
                                    veto=veto_mask(cw.sample(nbad, random_state=sd), T, N))["sr"])
        print(f"   무작위 {sd + 1:>2}: SR {plac[-1]:.2f}", flush=True)
    plac = np.array(plac)
    p80 = float(np.percentile(plac, 80))
    r2 = v1["sr"] > base["sr"] and v1["sr"] >= p80
    print(f"\n  ── R2 봇 ({START} 학습 시작, 검증 창 이어 붙임) ──")
    print(f"   우리 봇     SR {base['sr']:.2f}  누적 {base['tot']:+.1f}%  MDD {base['mdd']:.1f}%  거래 {base['n']}")
    print(f"   V1 매수거부 SR {v1['sr']:.2f}  누적 {v1['tot']:+.1f}%  MDD {v1['mdd']:.1f}%  거래 {v1['n']}")
    print(f"   무작위 거부 SR 평균 {plac.mean():.2f}, 범위 {plac.min():.2f}~{plac.max():.2f}, 80백분위 {p80:.2f}")

    print(f"\n  ── 사전 기준 ──")
    print(f"   R1 사건 20일 차이 {s['diff']:+.2f}%p, t={s['t']:+.2f} → {'✓' if r1 else '✗'}")
    print(f"   R2 V1 {v1['sr']:.2f} > 우리 봇 {base['sr']:.2f} 그리고 ≥ 무작위 80백분위 {p80:.2f} → {'✓' if r2 else '✗'}")
    verdict = "실전 도입 제안" if (r1 and r2) else ("약한 재현 — 그림자 기록 유지" if r2 else "재현 실패 — 아이디어 폐기")
    print(f"   ▶ 판정: {verdict}")


if __name__ == "__main__":
    main()
