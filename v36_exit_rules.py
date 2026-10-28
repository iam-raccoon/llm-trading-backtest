# -*- coding: utf-8 -*-
"""v36 — 청산 규칙 비교: 익절 / 본전 지키기 / 장중 손절. "이득 볼 땐 안 팔고 손해 보고서야 판다"에 대한 검증.

━━━ 왜 ━━━
실거래 MRNA: 매수 $193.20 → 종가 최고 $245.81(+27%) → 하루 −17.6%(종가 $202.63)로 손절선 $208 을
한 번에 뚫고 다음 날 시가 $190.48 매도(−1.6%). 사용자: 수익 중일 때 지키는 규칙이 필요하지 않나.

━━━ 변형 (우리 봇 = 돌파 진입 + 샹들리에, 분기 재선택 워크포워드, 슬롯 10, 비용 0.20%) ━━━
  TP10 / TP20 / TP30  종가 ≥ 진입가×(1+x) 이면 다음 시가 매도(익절)
  BE10                종가 최고가가 한 번이라도 진입가×1.10 이상이면 손절선을 max(샹들리에, 진입가)로
  INTRA               장중 손절: 전날까지 계산된 손절선에 그날 저가가 닿으면 그 값(시가가 이미 아래면 시가)에 매도
                      (스톱 주문 근사 — 체결 미끄러짐 없음 가정이라 낙관적이다)
각 변형은 그 규칙을 켠 채로 학습 창 파라미터도 다시 고른다.

━━━ 구간 (둘 다 판정에 쓸 수 있는 구간) ━━━
  A  2016-05 ~ 2021-03 (v35 주가, 처음 쓴 5년)       B  2022-09 ~ 2024-12 (v28 검증 창)
  봉인 구간(2025~)은 쓰지 않는다.

━━━ 사전 판정 기준 (결과 보기 전, 2026-10-29 커밋으로 고정) ━━━
  A·B **둘 다**에서 Sharpe ≥ 우리 봇 + 0.20, MDD 가 5%p 넘게 깊지 않음 → '실전 후보'(사용자 승인 후)
  INTRA 는 통과해도 토스 API 로 장중 스톱이 되는지 확인이 먼저다.
  그 외 → 기각. ⚠ 무작위로 매수 몇 번만 건너뛰어도 SR 이 ±0.4 출렁였다(v34b) — 0.2 는 최소 문턱.

사용: python3 v36_exit_rules.py
"""
import itertools

import numpy as np
import pandas as pd

import v28_walkforward as v28
import v35_8k_replicate as v35
from v26_us_holdout import build as build_b

SLOTS = 10
MARGIN, MDD_TOL = 0.20, 5.0
VARIANTS = {"TP10": {"tp": 0.10}, "TP20": {"tp": 0.20}, "TP30": {"tp": 0.30},
            "BE10": {"be": 0.10}, "INTRA": {"intra": True}}


def sim(ind, cfg, slots, i0, i1, tp=None, be=None, intra=False, cap=100.0):
    O, H, C = ind["op"].values, ind["hi"].values, ind["cl"].values
    L = ind["lo"].values
    A = ind["atr"].values
    D = ind["dc"][cfg["dc"]].values
    M = ind["ma"][cfg["ma"]].values
    R = ind["mom"][cfg["mom"]].values
    slot = cap / slots
    cash, pos, eq, tr = cap, {}, [], []
    for i in range(i0, min(i1, len(C) - 1)):
        for j in list(pos):
            p = pos[j]
            if intra and "stop" in p and not np.isnan(L[i, j]) and L[i, j] <= p["stop"]:
                px = min(O[i, j], p["stop"]) if O[i, j] > 0 else p["stop"]
                cash += p["q"] * px * (1 - v28.COST / 100)
                tr.append((px / p["e"] - 1) * 100 - v28.COST)
                del pos[j]
                continue
            if np.isnan(H[i, j]) or np.isnan(A[i, j]):
                continue
            p["pk"] = max(p["pk"], H[i, j])
            p["pkc"] = max(p.get("pkc", p["e"]), C[i, j])
            stop = p["pk"] - cfg["chand"] * A[i, j]
            if be is not None and p["pkc"] >= p["e"] * (1 + be):
                stop = max(stop, p["e"])
            p["stop"] = stop
            out = C[i, j] <= stop
            if tp is not None and C[i, j] >= p["e"] * (1 + tp):
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
            for j in np.argsort(-np.where(ok, R[i], -np.inf)):
                if len(pos) >= slots or cash < slot * 0.99 or not ok[j]:
                    break
                if j in pos:
                    continue
                cash -= slot
                pos[j] = {"q": slot / nx[j], "e": nx[j], "pk": nx[j]}
        eq.append(cash + sum(p["q"] * C[i, j] for j, p in pos.items() if not np.isnan(C[i, j])))
    return pd.Series(eq).pct_change().dropna().values, tr


def walkforward(ind, combos, idx, start, end, **kw):
    s0, s1 = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end))
    rets, trades, a = [], [], s0
    while a + v28.TRAIN + v28.TEST <= s1:
        best, bsr = None, -9e9
        for cfg in combos:
            sr = v28.sharpe(sim(ind, cfg, SLOTS, a, a + v28.TRAIN, **kw)[0])
            if sr > bsr:
                best, bsr = cfg, sr
        r, t = sim(ind, best, SLOTS, a + v28.TRAIN, a + v28.TRAIN + v28.TEST, **kw)
        rets.append(r)
        trades += t
        a += v28.TEST
    r = np.concatenate(rets)
    eq = (1 + pd.Series(r)).cumprod()
    return {"tot": (eq.iloc[-1] - 1) * 100, "sr": v28.sharpe(r),
            "mdd": float((eq / eq.cummax() - 1).min()) * 100, "n": len(trades),
            "win": 100 * float(np.mean(np.array(trades) > 0)) if trades else 0.0}


def with_low(ind, raw_px):
    import pickle
    raw = pickle.load(open(raw_px, "rb"))
    ind["lo"] = pd.DataFrame({c: raw[c]["df"]["Low"] for c in ind["codes"]}).sort_index() \
        .reindex(ind["cl"].index)
    return ind


def main():
    v28.COST = 0.20
    combos = [dict(zip(v28.GRID, v)) for v in itertools.product(*v28.GRID.values())]
    windows = {
        "A 2016~2021": (with_low(v35.build(), v35.PX), v35.START, v35.END),
        "B 2022~2024": (with_low(build_b(), "cache/v26_us_px.pkl"), v28.WF_START, v28.WF_END),
    }
    res = {}
    for wname, (ind, st, en) in windows.items():
        idx = ind["cl"].index
        # 정합성: 아무 규칙도 없으면 v28.sim 과 같아야 한다
        cfg = {"dc": 40, "ma": 120, "chand": 3.0, "mom": 20}
        s0 = idx.searchsorted(pd.Timestamp(st)) + 260
        r1, t1 = sim(ind, cfg, SLOTS, s0, s0 + 300)
        r2, t2 = v28.sim(ind, cfg, SLOTS, s0, s0 + 300)
        assert len(t1) == len(t2) and np.allclose(r1, r2), "시뮬레이터 불일치"
        res[wname] = {"우리 봇": walkforward(ind, combos, idx, st, en)}
        for k, kw in VARIANTS.items():
            res[wname][k] = walkforward(ind, combos, idx, st, en, **kw)
        print(f"\n[v36] {wname} (정합성 OK)")
        print(f"   {'':<10}{'누적%':>8}{'Sharpe':>8}{'MDD%':>8}{'거래':>6}{'승률%':>7}")
        for k, s in res[wname].items():
            print(f"   {k:<10}{s['tot']:>8.1f}{s['sr']:>8.2f}{s['mdd']:>8.1f}{s['n']:>6}{s['win']:>7.0f}")

    print("\n  ── 사전 기준 (A·B 둘 다 SR +0.20 이상, MDD 5%p 이내) ──")
    for k in VARIANTS:
        ok = all(res[w][k]["sr"] >= res[w]["우리 봇"]["sr"] + MARGIN
                 and res[w][k]["mdd"] >= res[w]["우리 봇"]["mdd"] - MDD_TOL for w in res)
        det = " / ".join(f"{w[:1]} {res[w][k]['sr']:.2f} vs {res[w]['우리 봇']['sr']:.2f}" for w in res)
        print(f"   {k:<8}{'실전 후보' if ok else '기각':<8}{det}")


if __name__ == "__main__":
    main()
