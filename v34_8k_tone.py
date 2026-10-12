# -*- coding: utf-8 -*-
"""v34 — 8-K 공시의 **단어 사전 점수**(Loughran-McDonald)가 미장 봇에 도움이 되나.

━━━ 왜 ━━━
사용자: "LLM 말고 단어만 보고 좋은 소식인지 판단해서 식에 넣자". 뉴스는 과거 시점 자료가 없어 검증이
안 되지만, 8-K 는 SEC EDGAR 에 접수 시각까지 남아 있어 백테스트가 된다. LLM 이 아니라 사전이라
같은 글에 항상 같은 점수가 나온다.

━━━ 데이터 ━━━
v34_8k_fetch.py → cache/v34_8k/scores.jsonl (587종목, 8-K 본문 + EX-99, 2021-06~2024-12).
점수 tone = (긍정어 − 부정어) ÷ 총 단어. 총 단어 200 미만 공시는 제외(인사 변경 등 짧은 공시).

━━━ 시점 (미래 정보 차단) ━━━
접수 시각(UTC→미 동부) 기준 '처음 거래 가능한 시가'에 연결한다.
  09:30 전 접수 → 그날 시가 / 09:30 이후·장 마감 후 접수 → 다음 거래일 시가.
봇의 판단일 i(종가로 판단, i+1 시가 체결)로 옮기면: 그 시가가 i+1 인 i.
'악재' 기준선 = 그 공시 **이전 252 거래일** 동안 접수된 공시들의 tone 하위 20% 값(시점별, 미래 미사용).

━━━ 확인 1: 사건 분석 ━━━
첫 거래 가능 시가 → +5, +20 거래일 종가 수익률에서 같은 기간 유니버스 동일가중 수익률을 뺀 초과수익.
악재(하위 20%) vs 나머지 평균 차이와 Welch t.

━━━ 확인 2: 봇 덧붙이기 (우리 봇, 슬롯 10, 분기 재선택 워크포워드, 2022-09~2024-12) ━━━
  V1 매수 거부: 판단일 기준 최근 5거래일 안에 악재 공시 → 신규 매수 안 함
  V2 악재 청산: 보유 종목에 판단일 악재 공시 → 다음 시가 매도
  V3 V1 + V2
학습 창의 파라미터 선택에도 같은 규칙을 적용한다(그 변형 자체를 재선택).

━━━ 사전 판정 기준 (결과 보기 전, 2026-10-12 커밋으로 고정) ━━━
 (A) 신호: 악재 − 나머지 20일 초과수익 평균 < 0 이고 Welch t ≤ −2.0
 (B) 봇: 변형 Sharpe ≥ 우리 봇 + 0.20, MDD 가 5%p 넘게 깊지 않음
 (A)와 (B)를 **둘 다** 만족하는 변형이 있으면(여럿이면 Sharpe 최고 하나) → '페이퍼 후보'.
 미장 한 시장뿐이므로 실전 교체 후보는 아니다. 그 외 → 기각.
 ⚠ 2.3년 상승장 구간, 생존편향 유니버스. Sharpe 0.3 미만 차이는 구분이 안 된다.

사용: python3 v34_8k_tone.py
"""
import itertools
import json

import numpy as np
import pandas as pd

import v28_walkforward as v28
from v26_us_holdout import build

SCORES = "/home/user/llm-trading-backtest/cache/v34_8k/scores.jsonl"
SLOTS = 10
MIN_WORDS = 200
LOOKBACK = 252
BAD_Q = 0.20
VETO_DAYS = 5
MARGIN, MDD_TOL, T_CUT = 0.20, 5.0, -2.0


def load_events(idx, codes):
    rows = [json.loads(x) for x in open(SCORES)]
    ev = pd.DataFrame(rows)
    ev = ev[(ev["words"] >= MIN_WORDS) & ev["ticker"].isin(codes)].copy()
    ev["tone"] = (ev["pos"] - ev["neg"]) / ev["words"]
    acc = pd.to_datetime(ev["accepted"], utc=True, errors="coerce")
    acc = acc.fillna(pd.to_datetime(ev["date"]).dt.tz_localize("UTC") + pd.Timedelta(hours=21))
    et = acc.dt.tz_convert("America/New_York")
    day = et.dt.tz_localize(None).dt.normalize()
    before_open = (et.dt.hour * 60 + et.dt.minute) < 9 * 60 + 30
    # 첫 거래 가능 시가의 위치 p: 장 전 접수면 그날(이상) 첫 거래일, 아니면 그 다음 거래일
    pos_same = idx.searchsorted(day.values, side="left")
    pos_next = idx.searchsorted(day.values, side="right")
    ev["open_pos"] = np.where(before_open.values, pos_same, pos_next)
    ev = ev[(ev["open_pos"] >= 1) & (ev["open_pos"] < len(idx))]
    ev["dec"] = ev["open_pos"] - 1                      # 봇 판단일 i (i+1 시가 체결)
    ev["col"] = ev["ticker"].map({c: k for k, c in enumerate(codes)})
    ev = ev.sort_values("open_pos").reset_index(drop=True)
    # 시점별 악재 기준선: 이 공시보다 앞서 '거래 가능해진' 지난 252일 공시들의 하위 20%
    thr = np.full(len(ev), np.nan)
    op = ev["open_pos"].values
    tone = ev["tone"].values
    for k in range(len(ev)):
        m = (op < op[k]) & (op >= op[k] - LOOKBACK)
        if m.sum() >= 200:
            thr[k] = np.quantile(tone[m], BAD_Q)
    ev["bad"] = tone <= thr
    ev["has_thr"] = ~np.isnan(thr)
    return ev


def event_study(ev, ind):
    O, C = ind["op"].values, ind["cl"].values
    out = {}
    for h in (5, 20):
        ar = []
        for p, j in zip(ev["open_pos"], ev["col"]):
            e = p + h - 1
            if e >= len(C) or not (O[p, j] > 0) or np.isnan(C[e, j]):
                ar.append(np.nan)
                continue
            r = C[e, j] / O[p, j] - 1
            with np.errstate(invalid="ignore", divide="ignore"):
                rb = np.nanmean(C[e] / O[p] - 1)      # 같은 시가→종가 구간의 동일가중 평균
            ar.append((r - rb) * 100)
        ev[f"ar{h}"] = ar
    e = ev[ev["has_thr"]]
    for h in (5, 20):
        a = e.loc[e["bad"], f"ar{h}"].dropna()
        b = e.loc[~e["bad"], f"ar{h}"].dropna()
        diff = a.mean() - b.mean()
        t = diff / np.sqrt(a.var() / len(a) + b.var() / len(b))
        out[h] = {"bad_n": len(a), "rest_n": len(b), "bad": a.mean(), "rest": b.mean(),
                  "diff": diff, "t": t}
    return out


def sim(ind, cfg, slots, i0, i1, veto=None, force=None, cap=100.0):
    """v28.sim 과 같은 회계 + (veto: 신규매수 금지 [T,N]) + (force: 다음 시가 청산 [T,N])."""
    C = ind["cl"].values
    O, H = ind["op"].values, ind["hi"].values
    A = ind["atr"].values
    D = ind["dc"][cfg["dc"]].values
    M = ind["ma"][cfg["ma"]].values
    R = ind["mom"][cfg["mom"]].values
    slot = cap / slots
    cash, pos, eq, tr = cap, {}, [], []
    for i in range(i0, min(i1, len(C) - 1)):
        for j in list(pos):
            p = pos[j]
            if np.isnan(H[i, j]) or np.isnan(A[i, j]):
                continue
            p["pk"] = max(p["pk"], H[i, j])
            out = C[i, j] <= p["pk"] - cfg["chand"] * A[i, j]
            if force is not None and force[i, j]:
                out = True
            if out:
                px = O[i + 1, j]
                if not (px > 0):
                    continue
                cash += p["q"] * px * (1 - v28.COST / 100)
                tr.append((px / p["e"] - 1) * 100 - v28.COST)
                del pos[j]
        if len(pos) < slots:
            nx = O[i + 1]
            ok = (C[i] > D[i]) & (C[i] > M[i]) & ~np.isnan(R[i]) & ~np.isnan(A[i]) & (nx > 0)
            if veto is not None:
                ok &= ~veto[i]
            for j in np.argsort(-np.where(ok, R[i], -np.inf)):
                if len(pos) >= slots or cash < slot * 0.99 or not ok[j]:
                    break
                if j in pos:
                    continue
                cash -= slot
                pos[j] = {"q": slot / nx[j], "e": nx[j], "pk": nx[j]}
        eq.append(cash + sum(p["q"] * C[i, j] for j, p in pos.items()
                             if not np.isnan(C[i, j])))
    e = pd.Series(eq)
    return e.pct_change().dropna().values, tr


def walkforward(ind, combos, idx, veto=None, force=None):
    s0 = idx.searchsorted(pd.Timestamp(v28.WF_START))
    s1 = idx.searchsorted(pd.Timestamp(v28.WF_END))
    rets, trades, a = [], [], s0
    while a + v28.TRAIN + v28.TEST <= s1:
        best, best_sr = None, -9e9
        for cfg in combos:
            r, _ = sim(ind, cfg, SLOTS, a, a + v28.TRAIN, veto, force)
            sr = v28.sharpe(r)
            if sr > best_sr:
                best, best_sr = cfg, sr
        r, tr = sim(ind, best, SLOTS, a + v28.TRAIN, a + v28.TRAIN + v28.TEST, veto, force)
        rets.append(r)
        trades += tr
        a += v28.TEST
    r = np.concatenate(rets)
    eq = (1 + pd.Series(r)).cumprod()
    return {"tot": (eq.iloc[-1] - 1) * 100, "sr": v28.sharpe(r),
            "mdd": float((eq / eq.cummax() - 1).min()) * 100, "n": len(trades)}


def main():
    ind = build()
    idx, codes = ind["cl"].index, ind["codes"]
    ev = load_events(idx, codes)
    print(f"[v34] 8-K 공시 {len(ev)}건 (200단어 이상, 종목 {ev['ticker'].nunique()}), "
          f"악재 판정 가능 {int(ev['has_thr'].sum())}건 중 악재 {int(ev['bad'].sum())}건")

    # 정합성: 빈 마스크면 v28.sim 과 같아야 한다
    cfg = {"dc": 40, "ma": 120, "chand": 3.0, "mom": 20}
    s0 = idx.searchsorted(pd.Timestamp("2022-09-01"))
    r1, t1 = sim(ind, cfg, SLOTS, s0, s0 + 300)
    r2, t2 = v28.sim(ind, cfg, SLOTS, s0, s0 + 300)
    same = len(t1) == len(t2) and np.allclose(r1, r2)
    print(f"  정합성(빈 마스크 = v28.sim): {same}")
    if not same:
        raise SystemExit("  ✗ 시뮬레이터 불일치")

    es = event_study(ev, ind)
    print("\n  ── 확인 1: 사건 분석 (시장 대비 초과수익 %, 첫 거래 가능 시가부터) ──")
    for h, s in es.items():
        print(f"   +{h:>2}일  악재 {s['bad']:+.2f} (n={s['bad_n']})  나머지 {s['rest']:+.2f} "
              f"(n={s['rest_n']})  차이 {s['diff']:+.2f}  t={s['t']:+.2f}")
    sig_a = es[20]["diff"] < 0 and es[20]["t"] <= T_CUT

    T, N = ind["cl"].shape
    bad_dec = np.zeros((T, N), bool)
    b = ev[ev["bad"]]
    bad_dec[b["dec"].values, b["col"].values] = True
    veto = np.zeros((T, N), bool)
    for k in range(VETO_DAYS):
        veto[k:] |= bad_dec[:T - k]

    combos = [dict(zip(v28.GRID, v)) for v in itertools.product(*v28.GRID.values())]
    res = {"우리 봇": walkforward(ind, combos, idx),
           "V1 매수 거부": walkforward(ind, combos, idx, veto=veto),
           "V2 악재 청산": walkforward(ind, combos, idx, force=bad_dec),
           "V3 둘 다": walkforward(ind, combos, idx, veto=veto, force=bad_dec)}
    print("\n  ── 확인 2: 봇 덧붙이기 (2022-09~2024-12 워크포워드) ──")
    print(f"   {'':<14}{'누적%':>8}{'Sharpe':>8}{'MDD%':>8}{'거래':>6}")
    for k, s in res.items():
        print(f"   {k:<14}{s['tot']:>8.1f}{s['sr']:>8.2f}{s['mdd']:>8.1f}{s['n']:>6}")

    base = res["우리 봇"]
    passed = [k for k in res if k != "우리 봇" and res[k]["sr"] >= base["sr"] + MARGIN
              and res[k]["mdd"] >= base["mdd"] - MDD_TOL]
    print(f"\n  ── 사전 기준 ──")
    print(f"   (A) 신호: 20일 차이 {es[20]['diff']:+.2f}%, t={es[20]['t']:+.2f} → {'✓' if sig_a else '✗'}")
    print(f"   (B) 봇 개선(SR +{MARGIN} 이상, MDD 5%p 이내): {passed or '없음'}")
    if sig_a and passed:
        pick = max(passed, key=lambda k: res[k]["sr"])
        print(f"   ▶ 판정: 페이퍼 후보 = {pick}")
    else:
        print(f"   ▶ 판정: 기각")


if __name__ == "__main__":
    main()
