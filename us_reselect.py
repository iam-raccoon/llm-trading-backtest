# -*- coding: utf-8 -*-
"""미장 봇 파라미터 **분기 재선택** — 직전 1년으로 고르고 설정파일에 쓴다.

━━━ 왜 고정하지 않나 ━━━
v28/v29 워크포워드에서 두 가지가 드러났다:
  · IS 에서 고른 설정의 성적(+177%)은 **고른 이득**이었고 공정하게 재면 +45.9% 였다.
  · 창마다 고른 설정이 dc20/40/60, ma50/120, ch2/3/4 로 **계속 바뀐다.**
    고정된 최적값이 없다. 하나를 박아두는 방식 자체가 불안정하다.
검증한 방식이 '분기마다 직전 1년으로 재선택'이었으므로 **그대로 굴린다.**
파라미터를 고정한 채 슬롯만 바꾸면 검증한 것과 다른 물건이 된다(v22 의 실패).

━━━ 절차 ━━━
  1. 유니버스 일봉을 갱신(FDR — 학습용이라 한 세션 지연은 무방)
  2. 직전 252 거래일에서 격자 36개를 훑어 **Sharpe 최고** 설정 선택
  3. `cache/us_trend_config.json` 에 기록(선택일·다음 재선택 예정일·이력 포함)

봇은 매 실행 때 이 파일을 읽는다. 파일이 없거나 90일 이상 묵으면 경고한다.

사용: python3 us_reselect.py            # 재선택 실행
      python3 us_reselect.py --show     # 현재 설정만 보기
"""
import argparse
import datetime as dt
import itertools
import json
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "cache", "us_trend_config.json")
PXC = os.path.join(HERE, "cache", "v26_us_px.pkl")

SLOTS = 10          # v28/v29 결과로 3 → 10 (두 구간 모두 슬롯3 이 꼴찌였다)
TRAIN = 252
REVIEW_DAYS = 90    # 분기
GRID = {"dc": [20, 40, 60], "ma": [50, 120], "chand": [2.0, 3.0, 4.0], "mom": [20, 60]}
COST = 0.20
ATR_N = 14


def load_config():
    if os.path.exists(CONFIG):
        try:
            return json.load(open(CONFIG, encoding="utf-8"))
        except Exception:
            pass
    return None


def refresh_prices():
    """학습용 일봉 갱신. 실패해도 기존 캐시로 진행한다."""
    import time

    import FinanceDataReader as fdr
    if not os.path.exists(PXC):
        raise RuntimeError("유니버스 캐시가 없다")
    raw = pickle.load(open(PXC, "rb"))
    since = (dt.date.today() - dt.timedelta(days=700)).isoformat()
    ok = 0
    for i, (s, v) in enumerate(raw.items()):
        try:
            d = fdr.DataReader(s, since).dropna(subset=["Close"])
            if len(d) > 300:
                v["df"] = d
                ok += 1
        except Exception:
            pass
        if (i + 1) % 150 == 0:
            print(f"    {i+1}/{len(raw)}", flush=True)
        time.sleep(0.02)
    pickle.dump(raw, open(PXC, "wb"))
    print(f"  가격 갱신 {ok}/{len(raw)}종목")
    return raw


def build(raw):
    codes = [c for c, v in raw.items() if len(v["df"]) > 300]
    f = lambda k: pd.DataFrame({c: raw[c]["df"][k] for c in codes}).sort_index()
    op, hi, lo, cl = f("Open"), f("High"), f("Low"), f("Close")
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo).stack(), (hi - pc).abs().stack(),
                    (lo - pc).abs().stack()], axis=1).max(axis=1).unstack()
    ind = {"op": op.values, "hi": hi.values, "cl": cl.values,
           "atr": tr.rolling(ATR_N).mean().values, "index": cl.index}
    ind["dc"] = {d: hi.rolling(d).max().shift(1).values for d in GRID["dc"]}
    ind["ma"] = {m: cl.rolling(m).mean().values for m in GRID["ma"]}
    ind["mom"] = {m: (cl / cl.shift(m) - 1).values for m in GRID["mom"]}
    return ind


def sim(ind, cfg, i0, i1, cap=100.0):
    O, H, C, A = ind["op"], ind["hi"], ind["cl"], ind["atr"]
    D, M, R = ind["dc"][cfg["dc"]], ind["ma"][cfg["ma"]], ind["mom"][cfg["mom"]]
    slot = cap / SLOTS
    cash, pos, eq = cap, {}, []
    for i in range(i0, min(i1, len(C) - 1)):
        for j in list(pos):
            p = pos[j]
            if np.isnan(H[i, j]) or np.isnan(A[i, j]):
                continue
            p["pk"] = max(p["pk"], H[i, j])
            if C[i, j] <= p["pk"] - cfg["chand"] * A[i, j]:
                px = O[i + 1, j]
                if px > 0:
                    cash += p["q"] * px * (1 - COST / 100)
                    del pos[j]
        if len(pos) < SLOTS:
            nx = O[i + 1]
            ok = (C[i] > D[i]) & (C[i] > M[i]) & ~np.isnan(R[i]) & ~np.isnan(A[i]) & (nx > 0)
            for j in np.argsort(-np.where(ok, R[i], -np.inf)):
                if len(pos) >= SLOTS or cash < slot * 0.99 or not ok[j]:
                    break
                if j in pos:
                    continue
                cash -= slot
                pos[j] = {"q": slot / nx[j], "e": nx[j], "pk": nx[j]}
        eq.append(cash + sum(p["q"] * C[i, j] for j, p in pos.items()
                             if not np.isnan(C[i, j])))
    e = pd.Series(eq)
    r = e.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 5 and r.std() else -9e9


def main(a):
    cur = load_config()
    if a.show:
        print(json.dumps(cur, ensure_ascii=False, indent=1) if cur else "설정 없음")
        return

    print("[재선택] 가격 갱신...", flush=True)
    raw = refresh_prices()
    ind = build(raw)
    n = len(ind["cl"])
    i0 = max(0, n - 1 - TRAIN)
    print(f"  학습 구간 {ind['index'][i0].date()} ~ {ind['index'][-1].date()} "
          f"({TRAIN}일) | 슬롯 {SLOTS}")

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    scored = sorted(((sim(ind, c, i0, n - 1), c) for c in combos), key=lambda x: -x[0])
    best_sr, best = scored[0]
    print(f"\n  상위 5:")
    for sr, c in scored[:5]:
        print(f"    SR {sr:5.2f}  dc{c['dc']} ma{c['ma']} ch{c['chand']} mom{c['mom']}")

    today = dt.date.today()
    out = {"slots": SLOTS, **best,
           "chosen_at": today.isoformat(),
           "train_from": str(ind["index"][i0].date()),
           "train_to": str(ind["index"][-1].date()),
           "train_sharpe": round(best_sr, 3),
           "next_review": (today + dt.timedelta(days=REVIEW_DAYS)).isoformat(),
           "history": (cur or {}).get("history", []) + (
               [{k: (cur or {}).get(k) for k in
                 ("chosen_at", "dc", "ma", "chand", "mom", "slots", "train_sharpe")}]
               if cur else [])}
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    json.dump(out, open(CONFIG, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  선정: dc{best['dc']} ma{best['ma']} ch{best['chand']} mom{best['mom']} "
          f"슬롯{SLOTS}  (학습 SR {best_sr:.2f})")
    print(f"  다음 재선택 {out['next_review']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    main(ap.parse_args())
