# -*- coding: utf-8 -*-
"""v35 주가 수집 — 미장 유니버스(v26 와 같은 587종목)의 2014-06 ~ 2021-06 일봉.
v26_us_px.pkl 과 같은 모양({sym: {"name", "df"}})으로 cache/v35_us_px.pkl 에 저장.
⚠ 유니버스가 '지금' 구성종목이라 2016 년 시점으로는 생존편향이 더 크다(비교 대상 모두 같은 조건)."""
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import FinanceDataReader as fdr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "cache", "v26_us_px.pkl")
OUT = os.path.join(HERE, "cache", "v35_us_px.pkl")


def one(s, name):
    for k in range(3):
        try:
            d = fdr.DataReader(s, "2014-06-01", "2021-06-30").dropna(subset=["Close"])
            if len(d) > 300:
                return s, {"name": name, "df": d}
            return None
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def main():
    src = pickle.load(open(SRC, "rb"))
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(one, s, v["name"]) for s, v in src.items()]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                out[r[0]] = r[1]
            if k % 100 == 0:
                print(f"  {k}/{len(src)}  확보 {len(out)}  {time.time() - t0:.0f}초", flush=True)
    pickle.dump(out, open(OUT, "wb"))
    print(f"[v35] 주가 {len(out)}/{len(src)}종목 저장 → {OUT}", flush=True)


if __name__ == "__main__":
    main()
