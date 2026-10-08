# -*- coding: utf-8 -*-
"""한투 ②모멘텀 — 미장 **포워드 페이퍼**. 주문은 절대 내지 않는다.

━━━ 왜 ━━━
v30: 2022-09~2024 에서 미장 SR 1.37 · 국장 SR 1.40 으로 두 시장 모두 시장을 이겼다.
v31: 봉인 구간(2025-01~2026-07)에서 SR 0.55 · MDD −41% 로 **사전 기준 불통과**.
기준대로 실전은 지금 봇을 유지하고, ②는 여기서 페이퍼로만 지켜본다.

━━━ 규칙 (원본 그대로, v30.run 과 같은 회계) ━━━
    신호   전 세션 종가 기준 60일 수익률 r60
    매수   r60 ≥ +30%, 강도 min(1, 0.5+r60) 순, 동점은 유니버스 순서
    매도   보유 종목 r60 ≤ −20%
    체결   다음 세션 **시가**. 슬롯 10, 슬롯 금액 = 시작자본/10 고정(백테스트와 같음)
    비용   매도 때 0.20%

━━━ 언제 도나 ━━━
한국 07:00(미국 장 마감 직후). 완료된 세션만 처리하므로 백테스트와 똑같이 재현된다.
⚠️ 실전 봇(22:40·23:40·00:40)과 시간을 겹치지 말 것 — 토스는 토큰을 새로 받으면
이전 토큰을 무효화한다. 놓친 날이 있으면 다음 실행 때 밀린 세션을 차례로 처리한다.

사용: python3 us_mom60_paper.py            # 밀린 세션 처리
      python3 us_mom60_paper.py --status   # 현황만
"""
import argparse
import datetime as dt
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

import pandas as pd

from us_trend_paper import ny_now, symbols, toss_candles

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "cache", "us_mom60_state.json")
CAP, SLOTS, COST = 72.5, 10, 0.20
BUY_TH, SELL_TH, LOOK = 0.30, -0.20, 60
DEADLINE = 600
MIN_COVER = 0.90


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE, encoding="utf-8"))
    return {"start": None, "cash": CAP, "pos": {}, "closed": [], "equity": [],
            "last_date": None}


def save(s):
    tmp = STATE + ".tmp"
    json.dump(s, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, STATE)


def fetch(t, syms):
    out = {}
    ex = ThreadPoolExecutor(max_workers=6)
    futs = {ex.submit(toss_candles, t, s): s for s in syms}
    try:
        for fu in as_completed(futs, timeout=DEADLINE):
            try:
                d = fu.result(timeout=1)
            except Exception:
                continue
            if d is not None and len(d) > LOOK + 2:
                out[futs[fu]] = d
    except TimeoutError:
        print(f"  ⚠️ 마감 {DEADLINE}초 — {len(out)}/{len(syms)}")
    for fu in futs:
        fu.cancel()
    ex.shutdown(wait=False)
    return out


def completed_cutoff():
    """이 날짜 **이하**의 봉은 완성된 것이다. 16:30 ET 이후면 오늘 봉도 완성."""
    now = ny_now()
    d = now.date()
    return d if now.time() >= dt.time(16, 30) else d - dt.timedelta(days=1)


def status(s):
    eq = s["equity"][-1][1] if s["equity"] else CAP
    print(f"  시작 {s['start']} | 마지막 처리 세션 {s['last_date']} | 평가 ${eq:.2f} "
          f"({(eq / CAP - 1) * 100:+.2f}%) | 현금 ${s['cash']:.2f}")
    for k, p in s["pos"].items():
        print(f"   · {k:<6} {p['qty']:.6f}주 @${p['entry']:.2f} ({p['date']})")
    if s["closed"]:
        rs = [c["ret"] for c in s["closed"]]
        print(f"  청산 {len(rs)}건 · 평균 {sum(rs) / len(rs):+.2f}% · "
              f"승률 {100 * sum(r > 0 for r in rs) / len(rs):.0f}%")


def main(a):
    s = load()
    print(f"===== ②모멘텀 페이퍼 {dt.date.today()} =====")
    if a.status:
        status(s)
        return

    from toss_trade import Toss
    t = Toss()
    syms = [c for c, _ in symbols()]
    t0 = time.time()
    data = fetch(t, syms)
    print(f"  조회 {len(data)}/{len(syms)} ({time.time() - t0:.0f}초)")
    if len(data) < len(syms) * MIN_COVER:
        print(f"  ⚠️ 조회 {MIN_COVER:.0%} 미만 — 신호 왜곡 우려로 오늘은 처리하지 않는다")
        return

    order = [c for c in syms if c in data]          # 동점 순서 = 유니버스 순서
    O = pd.DataFrame({c: data[c]["Open"] for c in order}).sort_index()
    C = pd.DataFrame({c: data[c]["Close"] for c in order}).sort_index()
    cut = pd.Timestamp(completed_cutoff())
    O, C = O[O.index <= cut], C[C.index <= cut]
    r60 = C.pct_change(LOOK, fill_method=None)
    dates = list(C.index)

    if s["last_date"] is None:
        # 첫 실행: 가장 최근 완료 세션 하나만 처리하고 시작한다
        s["start"] = str(dates[-1].date())
        s["last_date"] = str(dates[-2].date())
    todo = [k for k, d in enumerate(dates)
            if d > pd.Timestamp(s["last_date"]) and k >= LOOK + 1]
    if not todo:
        print(f"  처리할 새 세션 없음(마지막 {s['last_date']})")
        status(s)
        return

    slot = CAP / SLOTS
    for k in todo:
        d, prev = dates[k], k - 1
        day = str(d.date())
        rp = r60.iloc[prev]
        op = O.iloc[k]
        for sym in list(s["pos"]):
            p = s["pos"][sym]
            x = rp.get(sym)
            px = op.get(sym)
            if x is None or pd.isna(x) or not (px and px > 0):
                continue
            if x <= SELL_TH:
                s["cash"] += p["qty"] * px * (1 - COST / 100)
                ret = (px / p["entry"] - 1) * 100 - COST
                s["closed"].append({"sym": sym, "in": p["date"], "out": day,
                                    "ret": round(ret, 2)})
                print(f"  ▣ {day} 매도 {sym} @${px:.2f} ({ret:+.2f}%, r60 {x * 100:+.0f}%)")
                del s["pos"][sym]
        if len(s["pos"]) < SLOTS:
            cands = []
            for i, sym in enumerate(order):
                x, px = rp.get(sym), op.get(sym)
                if sym in s["pos"] or pd.isna(x) or not (px and px > 0) or x < BUY_TH:
                    continue
                cands.append((-min(1.0, 0.5 + x), i, sym, px, x))
            for _, _, sym, px, x in sorted(cands):
                if len(s["pos"]) >= SLOTS or s["cash"] < slot * 0.99:
                    break
                s["cash"] -= slot
                s["pos"][sym] = {"qty": slot / px, "entry": float(px), "date": day}
                print(f"  ▶ {day} 매수 {sym} ${slot:.2f} @${px:.2f} (r60 {x * 100:+.0f}%)")
        cl = C.iloc[k]
        mv = s["cash"] + sum(p["qty"] * (cl.get(sym) if pd.notna(cl.get(sym))
                                         else p["entry"]) for sym, p in s["pos"].items())
        s["equity"].append([day, round(float(mv), 4)])
        s["last_date"] = day
        save(s)
    status(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    main(ap.parse_args())
