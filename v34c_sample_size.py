# -*- coding: utf-8 -*-
"""v34c — 그림자 기록을 '얼마나' 해야 하나. 2022-09~2024-12 돌파 후보 기준으로
(1) 악재 표시가 붙는 후보의 빈도, (2) 표시/비표시 후보의 20일 초과수익 차이를 재서
t = −2 에 필요한 표본 수와 기간을 추정한다. 계획용 사후 추정(판정 아님)."""
import numpy as np
import pandas as pd

import v34_8k_tone as v34
from v26_us_holdout import build

CFG = {"dc": 40, "ma": 50, "mom": 60}      # 현재 실전 설정


def main():
    ind = build()
    idx, codes = ind["cl"].index, ind["codes"]
    ev = v34.load_events(idx, codes)
    T, N = ind["cl"].shape
    bad = np.zeros((T, N), bool)
    b = ev[ev["bad"]]
    bad[b["dec"].values, b["col"].values] = True
    flag = np.zeros((T, N), bool)
    for k in range(v34.VETO_DAYS):
        flag[k:] |= bad[:T - k]

    C, O = ind["cl"].values, ind["op"].values
    D = ind["dc"][CFG["dc"]].values
    M = ind["ma"][CFG["ma"]].values
    cand = (C > D) & (C > M)
    i0 = idx.searchsorted(pd.Timestamp("2022-09-01"))
    i1 = idx.searchsorted(pd.Timestamp("2024-12-03"))
    rows = []
    for i in range(i0, i1 - 21):
        js = np.where(cand[i])[0]
        if not len(js):
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            mk = np.nanmean(C[i + 20] / O[i + 1] - 1)
        for j in js:
            if O[i + 1, j] > 0 and not np.isnan(C[i + 20, j]):
                rows.append((i, flag[i, j], (C[i + 20, j] / O[i + 1, j] - 1 - mk) * 100))
    df = pd.DataFrame(rows, columns=["i", "flag", "ar20"])
    days = df["i"].nunique()
    per_day = len(df) / days
    f = df[df["flag"]]["ar20"]
    u = df[~df["flag"]]["ar20"]
    diff = f.mean() - u.mean()
    sd = df["ar20"].std()
    print(f"  돌파 후보: 하루 평균 {per_day:.1f}개, 그중 악재 표시 비율 {df['flag'].mean() * 100:.1f}%")
    print(f"  20일 초과수익: 표시 {f.mean():+.2f}% (n={len(f)}) / 비표시 {u.mean():+.2f}% (n={len(u)})"
          f" / 차이 {diff:+.2f}%p, 표준편차 {sd:.1f}%")
    # 같은 날 후보끼리는 겹쳐서 독립이 아니다 → 하루 단위 유효 표본은 '종목-일'보다 훨씬 적다.
    # 보수적으로 같은 종목의 연속 표시를 한 사건으로 친다(20일 보유가 겹치므로).
    ff = df[df["flag"]].copy()
    ff["j"] = 0
    if abs(diff) > 0:
        n_need = (2 * sd / abs(diff)) ** 2
        flags_per_year = df["flag"].sum() / (days / 252)
        print(f"  t=−2 에 필요한 표시 후보 ≈ {n_need:.0f}개 (종목-일 기준)")
        print(f"  표시 후보가 1년에 약 {flags_per_year:.0f}개(종목-일) 나오므로 → 약 "
              f"{n_need / flags_per_year * 12:.0f}개월 (겹침 때문에 실제로는 더 걸림)")


if __name__ == "__main__":
    main()
