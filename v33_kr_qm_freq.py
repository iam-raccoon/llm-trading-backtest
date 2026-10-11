# -*- coding: utf-8 -*-
"""v33 — 국장 퀀트 모멘텀(Gray & Vogel)을 교체 주기별로 비교. 분기 원본 vs 매일 거래.

━━━ 왜 ━━━
v32 에서 퀀트 모멘텀(분기 교체)이 국장 2022-09~2024-09 에서 SR 1.21(+94%)로 우리 봇 방식(0.49)·
시장(0.78)을 모두 이겼다. 사용자 요청: "하루 한 번 거래로 해서 비교". 미장 봇처럼 매일 확인한다.

━━━ 변형 (선정 규칙은 v32.quant_momentum 과 같음: 12-2 상위 10% → ID 낮은 절반 → ID 낮은 10) ━━━
  Q  분기(원본)  2·5·8·11월 마지막 거래일 교체
  M  월          매월 마지막 거래일 교체
  W  주          매주 마지막 거래일 교체
  D  매일(엄격)  매일, 목표 10종목에서 빠지면 매도
  DB 매일(완충)  매일, 보유 종목은 '상위 10% ∩ ID 낮은 절반' 밖으로 밀려날 때만 매도, 빈 칸은 매일 채움
비용 0.23%(왕복), 슬롯 10, 다음 날 시가 체결. ⚠ 국장은 1주 단위인데 시뮬은 소수점(금액) 체결이다.

━━━ 구간 ━━━
  선택 구간 2022-09-13 ~ 2024-09-30 (v32 와 같음)
  확인 구간 2024-10-01 ~ 2026-08-28 (데이터 끝). 퀀트 모멘텀은 외부 고정 규칙이라 이 구간을 처음
  쓰는 셈이다. 국장 봉인 구간은 우리 봇 기준으로는 이미 여러 번 열렸다는 점은 밝혀 둔다.

━━━ 사전 판정 기준 (결과 보기 전, 2026-10-11 커밋으로 고정) ━━━
 1. 선택 구간에서 Sharpe 가 가장 높은 변형 **하나**를 고른다(다른 구간 결과는 고르는 데 안 쓴다).
 2. 그 변형이 확인 구간에서 (a) Sharpe > 시장(동일가중) 그리고 (b) Sharpe ≥ 우리 봇(분기 재선택 WF)
    이면 '국장 페이퍼 후보'. 아니면 기각.
 3. 확인 구간에 다른 변형 숫자도 찍지만 판정에는 쓰지 않는다(참고).

사용: python3 v33_kr_qm_freq.py
"""
import itertools

import numpy as np
import pandas as pd

import v28_walkforward as v28
import v30_kis_strategies as v30

SLOTS = 10
SEL = ("2022-09-13", "2024-09-30")
CHK = ("2024-10-01", "2026-08-28")


def qm_parts(cl):
    m = cl.shift(21) / cl.shift(252) - 1
    dr = cl.pct_change(fill_method=None)
    pos = (dr > 0).astype(float).shift(21).rolling(231).mean()
    neg = (dr < 0).astype(float).shift(21).rolling(231).mean()
    idisc = np.sign(m) * (neg - pos)
    valid = m.notna() & idisc.notna()
    dec = m.where(valid).rank(axis=1, pct=True, ascending=False) <= 0.10
    half = idisc.where(dec).rank(axis=1, pct=True, ascending=True) <= 0.5
    idr = idisc.where(dec).rank(axis=1, ascending=True)
    keep = (dec & half).fillna(False)
    target = (keep & (idr <= SLOTS)).fillna(False)
    return idisc, keep, target


def reb_mask(idx, freq):
    d = pd.Series(idx, index=idx)
    if freq == "D":
        return np.ones(len(idx), bool)
    if freq == "W":
        key = [idx.isocalendar().year, idx.isocalendar().week]
    else:
        key = [idx.year, idx.month]
    last = (d.groupby(key).transform("max") == d).values
    if freq == "Q":
        last &= idx.month.isin([2, 5, 8, 11])
    return last


def signals(ind, freq):
    cl = ind["cl"]
    idisc, keep, target = qm_parts(cl)
    rb = reb_mask(cl.index, "D" if freq == "DB" else freq)[:, None]
    buy = rb & target.values
    if freq == "DB":
        sell = rb & ~keep.values
    else:
        sell = rb & ~target.values
    return buy, (-idisc).values, sell


def wf_series(ind, combos, idx, start, end):
    """우리 봇 워크포워드 — 첫 검증창이 start 에서 시작, end 까지."""
    stop = idx.searchsorted(pd.Timestamp(end), side="right")
    a = idx.searchsorted(pd.Timestamp(start)) - v28.TRAIN
    rets, trades = [], []
    while a + v28.TRAIN < stop:
        e = min(a + v28.TRAIN + v28.TEST, stop)
        best, best_sr = None, -9e9
        for cfg in combos:
            r, _ = v28.sim(ind, cfg, SLOTS, a, a + v28.TRAIN)
            sr = v28.sharpe(r)
            if sr > best_sr:
                best, best_sr = cfg, sr
        r, tr = v28.sim(ind, best, SLOTS, a + v28.TRAIN, e)
        rets.append(r)
        trades += tr
        a += v28.TEST
    return np.concatenate(rets), trades


def main():
    v30.COST = v28.COST = 0.23
    ind = v30.build_kr()
    idx = ind["cl"].index
    combos = [dict(zip(v28.GRID, v)) for v in itertools.product(*v28.GRID.values())]
    names = {"Q": "분기(원본)", "M": "월", "W": "주", "D": "매일(엄격)", "DB": "매일(완충)"}
    sig = {f: signals(ind, f) for f in names}

    res = {}
    for tag, (lo, hi) in (("선택", SEL), ("확인", CHK)):
        a0 = idx.searchsorted(pd.Timestamp(lo))
        a1 = idx.searchsorted(pd.Timestamp(hi), side="right")
        rows = {}
        r, tr = wf_series(ind, combos, idx, lo, hi)
        rows["우리 봇(분기 재선택)"] = v30.stats(r, tr)
        b = ind["bench"].iloc[a0:a1]
        rows["시장(동일가중)"] = v30.stats(b.pct_change().dropna().values, [])
        for f, nm in names.items():
            rows[f"QM {nm}"] = v30.stats(*v30.run(ind, *sig[f], a0, a1))
        res[tag] = rows
        print(f"\n[v33] 국장 {tag} 구간 {idx[a0].date()} ~ {idx[a1 - 1].date()} (비용 0.23%)")
        print(f"  {'':<24}{'누적%':>8}{'Sharpe':>8}{'MDD%':>8}{'거래':>6}{'연환산 거래':>10}")
        print("  " + "-" * 64)
        yrs = (a1 - a0) / 252
        for k, s in rows.items():
            print(f"  {k:<24}{s['tot']:>8.1f}{s['sr']:>8.2f}{s['mdd']:>8.1f}{s['n']:>6}"
                  f"{s['n'] / yrs:>10.0f}")

    qm = {k: v for k, v in res["선택"].items() if k.startswith("QM")}
    pick = max(qm, key=lambda k: qm[k]["sr"])
    c = res["확인"]
    ok_a = c[pick]["sr"] > c["시장(동일가중)"]["sr"]
    ok_b = c[pick]["sr"] >= c["우리 봇(분기 재선택)"]["sr"]
    print(f"\n  ── 사전 기준 ──")
    print(f"  선택 구간 최고 = {pick} (SR {qm[pick]['sr']:.2f})")
    print(f"  확인 구간: (a) SR > 시장   {c[pick]['sr']:.2f} vs {c['시장(동일가중)']['sr']:.2f}  "
          f"{'✓' if ok_a else '✗'}")
    print(f"             (b) SR ≥ 우리 봇 {c[pick]['sr']:.2f} vs {c['우리 봇(분기 재선택)']['sr']:.2f}  "
          f"{'✓' if ok_b else '✗'}")
    print(f"  ▶ 판정: {'국장 페이퍼 후보' if ok_a and ok_b else '기각'}")


if __name__ == "__main__":
    main()
