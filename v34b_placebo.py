# -*- coding: utf-8 -*-
"""v34b — V1(악재 공시 뒤 신규 매수 거부)의 SR 1.94 가 우연인지 보는 **위약(placebo) 검사**. 사후 점검.

v34 판정은 이미 '기각'(사건 분석에서 악재 공시의 예측력이 없음, 20일 t=−0.51).
그런데 봇 변형 V1 이 SR 1.43 → 1.94 로 크게 올랐다. 예측력 없는 신호로 좋아졌다면 슬롯 10개에서
몇 번 매수를 건너뛴 탓에 산 종목이 바뀐 '나비 효과'일 수 있다.
같은 수의 공시를 **무작위로** 악재로 찍어 똑같이 매수 거부를 걸었을 때 SR 분포를 본다.
판정을 바꾸는 용도가 아니다.

사용: python3 v34b_placebo.py
"""
import itertools

import numpy as np

import v28_walkforward as v28
import v34_8k_tone as v34
from v26_us_holdout import build

SEEDS = 20


def main():
    ind = build()
    idx, codes = ind["cl"].index, ind["codes"]
    ev = v34.load_events(idx, codes)
    elig = ev[ev["has_thr"]]
    nbad = int(elig["bad"].sum())
    T, N = ind["cl"].shape
    combos = [dict(zip(v28.GRID, v)) for v in itertools.product(*v28.GRID.values())]

    def veto_from(rows):
        bd = np.zeros((T, N), bool)
        bd[rows["dec"].values, rows["col"].values] = True
        vt = np.zeros((T, N), bool)
        for k in range(v34.VETO_DAYS):
            vt[k:] |= bd[:T - k]
        return vt

    real = v34.walkforward(ind, combos, idx, veto=veto_from(elig[elig["bad"]]))
    srs = []
    for s in range(SEEDS):
        pick = elig.sample(nbad, random_state=s)
        srs.append(v34.walkforward(ind, combos, idx, veto=veto_from(pick))["sr"])
        print(f"  무작위 {s + 1:>2}: SR {srs[-1]:.2f}", flush=True)
    srs = np.array(srs)
    print(f"\n  실제 악재 거부 SR {real['sr']:.2f} | 무작위 거부 SR 평균 {srs.mean():.2f}, "
          f"범위 {srs.min():.2f}~{srs.max():.2f}, 실제 이상인 비율 {np.mean(srs >= real['sr']) * 100:.0f}%")
    print(f"  (아무것도 안 거른 우리 봇 SR 1.43)")


if __name__ == "__main__":
    main()
