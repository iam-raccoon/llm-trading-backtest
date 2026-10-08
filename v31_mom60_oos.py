# -*- coding: utf-8 -*-
"""v31 — 한투 ②모멘텀(60일 +30% 매수 / −20% 매도)을 **미장 봉인 구간**에서 한 번 확인.

━━━ 왜 여는가 ━━━
v30 에서 한투 프리셋 10종 중 ②만 미장(SR 1.37)·국장(SR 1.40) **두 시장 모두**에서
시장을 이겼다. 파라미터는 한투가 정한 기본값이라 우리가 이 데이터에 맞춰 고른 게 아니다.
그래서 봉인 구간에서 한 번 재는 것이 정당하다. 이 구간은 v26·v29 에 이어 **세 번째**다.
이후 다시 닫는다. 다음 증거는 포워드 페이퍼(us_mom60_paper.py)에서만 나온다.

━━━ 사전 판정 기준 (결과 보기 전, 2026-10-08 커밋으로 고정) ━━━
판정 대상은 **②원본 하나**(한투 기본값, 추적손절 없음).
비교 상대는 **우리 봇(슬롯 10, 분기 재선택 워크포워드)**, 같은 날짜 구간.
아래 네 가지를 **모두** 만족해야 '통과':
  (1) ② Sharpe ≥ 우리 봇 Sharpe
  (2) ② MDD 가 우리 봇보다 5%p 넘게 깊지 않음
  (3) ② Sharpe > 시장(유니버스 동일가중) Sharpe
  (4) ② 청산 거래 ≥ 15건 — 미만이면 '판정 불가'(표본이 작아 운과 구분 안 됨)
통과 → 실전 교체 **후보**로 사용자에게 제안(자동 교체 안 함), 페이퍼는 계속.
불통과/판정 불가 → 실전은 지금 봇 유지, ②는 페이퍼로만 지켜본다.
②+샹들리에는 v30 에서 결과를 본 뒤 붙인 조합이므로 **참고로만** 표시하고 판정에 안 쓴다.

사용: python3 v31_mom60_oos.py
"""
import itertools

import numpy as np
import pandas as pd

import v30_kis_strategies as v30
from v26_us_holdout import build
from v28_walkforward import GRID, TEST, TRAIN, sharpe
from v29_oos_walkforward import OOS_FROM, OOS_TO, walkforward_oos

MDD_TOL = 5.0
MIN_TRADES = 15


def main():
    ind = build()
    idx = ind["cl"].index

    # v29 와 정확히 같은 검증 창 범위
    stop = idx.searchsorted(pd.Timestamp(OOS_TO))
    a0 = idx.searchsorted(pd.Timestamp(OOS_FROM))
    a = a0 - TRAIN
    while a + TRAIN + TEST <= stop:
        a += TEST
    a1 = a + TRAIN
    print(f"[v31] 봉인 구간 {idx[a0].date()} ~ {idx[a1 - 1].date()} — ②모멘텀 vs 우리 봇 (세 번째 개봉)\n")

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    wf = walkforward_oos(ind, v30.SLOTS, combos, idx)

    b = ind["bench"].iloc[a0:a1]
    br = b.pct_change().dropna().values
    mkt = v30.stats(br, [])

    cl = ind["cl"]
    r60 = cl.pct_change(60, fill_method=None)
    buy = (r60 >= 0.30).fillna(False).values
    score = (0.5 + r60).clip(upper=1.0).values
    sell = (r60 <= -0.20).fillna(False).values
    m = v30.stats(*v30.run(ind, buy, score, sell, a0, a1))
    mc = v30.stats(*v30.run(ind, buy, score, sell, a0, a1, chand=3.0))

    print(f"  {'':<30}{'누적%':>8}{'Sharpe':>8}{'MDD%':>8}{'거래':>6}")
    print("  " + "-" * 60)
    for name, s in (("우리 봇(분기 재선택 WF)", wf), ("시장(동일가중)", mkt),
                    ("②모멘텀 원본  ← 판정 대상", m), ("②+샹들리에 (참고만)", mc)):
        print(f"  {name:<30}{s['tot']:>8.1f}{s['sr']:>8.2f}{s['mdd']:>8.1f}{s['n']:>6}")

    c1 = m["sr"] >= wf["sr"]
    c2 = m["mdd"] >= wf["mdd"] - MDD_TOL
    c3 = m["sr"] > mkt["sr"]
    c4 = m["n"] >= MIN_TRADES
    print(f"\n  사전 기준")
    print(f"   (1) Sharpe ≥ 우리 봇       {m['sr']:.2f} vs {wf['sr']:.2f}  {'✓' if c1 else '✗'}")
    print(f"   (2) MDD 5%p 이내           {m['mdd']:.1f} vs {wf['mdd']:.1f}  {'✓' if c2 else '✗'}")
    print(f"   (3) Sharpe > 시장          {m['sr']:.2f} vs {mkt['sr']:.2f}  {'✓' if c3 else '✗'}")
    print(f"   (4) 거래 ≥ {MIN_TRADES}               {m['n']}  {'✓' if c4 else '✗'}")
    if not c4:
        verdict = "판정 불가 — 거래가 너무 적다. 실전 유지, ②는 페이퍼로만"
    elif c1 and c2 and c3:
        verdict = "통과 — 실전 교체 '후보'. 사용자 승인 전엔 바꾸지 않는다"
    else:
        verdict = "불통과 — 실전은 지금 봇 유지, ②는 페이퍼로만"
    print(f"\n  ▶ 판정: {verdict}")
    print(f"  ※ 이 구간은 다시 닫는다. 다음 증거는 포워드 페이퍼뿐이다.")


if __name__ == "__main__":
    main()
