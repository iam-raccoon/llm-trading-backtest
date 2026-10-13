# -*- coding: utf-8 -*-
"""v35 수집기 — 2015-06 ~ 2021-05 8-K (v34 와 겹치지 않는 과거 5년).

v34 와 같은 사전·같은 점수식. 다른 점: 본문은 **필요한 공시만** 받는다.
  · 후보 창 공시: 그 종목이 (어떤 격자 설정으로든) 돌파 후보였던 판단일 i 에 대해
    판단일 기준 i−4..i 에 걸리는 공시 → 매수 거부 판정에 실제로 쓰인다.
  · 무작위 30% 표본: 접수번호 해시로 정한 30% (후보 여부와 무관) → 시점별 '하위 20%' 기준값을
    편향 없이 추정하는 데만 쓴다.
사용: python3 v35_8k_fetch.py list   # 공시 목록(meta.jsonl)
      python3 v35_8k_fetch.py score  # 필요한 공시만 점수(scores.jsonl), 이어 받기 가능
"""
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import v34_8k_fetch as f

f.D0, f.D1 = "2015-06-01", "2021-05-31"
f.GAP = 0.12
HERE = os.path.dirname(os.path.abspath(__file__))
PX = os.path.join(HERE, "cache", "v35_us_px.pkl")
DIR = "/home/user/llm-trading-backtest/cache/v35_8k"
META = os.path.join(DIR, "meta.jsonl")
OUT = os.path.join(DIR, "scores.jsonl")
SAMPLE = 0.30
WORKERS = 8


def sampled(acc):
    return int(hashlib.md5(acc.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < SAMPLE


def do_list():
    os.makedirs(DIR, exist_ok=True)
    tickers = list(pickle.load(open(os.path.join(HERE, "cache", "v26_us_px.pkl"), "rb")).keys())
    cmap = f.ticker_ciks(tickers)
    done = set()
    if os.path.exists(META):
        done = {json.loads(x)["ticker"] for x in open(META)}
    t0 = time.time()
    with open(META, "a") as fo:
        for k, (tk, cik) in enumerate(cmap.items(), 1):
            if tk in done:
                continue
            try:
                rows = f.filings_8k(cik)
            except Exception as e:
                print(f"  ⚠ {tk} {e}", flush=True)
                continue
            for r in rows:
                fo.write(json.dumps({"ticker": tk, "cik": cik, **r}) + "\n")
            fo.flush()
            if k % 50 == 0:
                print(f"  목록 {k}/{len(cmap)} 종목, {time.time() - t0:.0f}초", flush=True)
    n = sum(1 for _ in open(META))
    print(f"[v35] 공시 목록 {n}건", flush=True)


def first_open_pos(meta, idx):
    acc = pd.to_datetime(meta["accepted"], utc=True, errors="coerce")
    acc = acc.fillna(pd.to_datetime(meta["date"]).dt.tz_localize("UTC") + pd.Timedelta(hours=21))
    et = acc.dt.tz_convert("America/New_York")
    day = et.dt.tz_localize(None).dt.normalize()
    before = (et.dt.hour * 60 + et.dt.minute) < 9 * 60 + 30
    ps = idx.searchsorted(day.values, side="left")
    pn = idx.searchsorted(day.values, side="right")
    return np.where(before.values, ps, pn)


def needed(meta):
    raw = pickle.load(open(PX, "rb"))
    codes = list(raw)
    cl = pd.DataFrame({c: raw[c]["df"]["Close"] for c in codes}).sort_index()
    hi = pd.DataFrame({c: raw[c]["df"]["High"] for c in codes}).sort_index()
    idx = cl.index
    dc20 = hi.rolling(20).max().shift(1)
    ma = np.minimum(cl.rolling(50).mean(), cl.rolling(120).mean())
    cand = ((cl > dc20) & (cl > ma)).values                  # 어떤 격자 설정으로든 후보일 수 있는 상위집합
    col = {c: k for k, c in enumerate(codes)}
    p = first_open_pos(meta, idx)
    need_c = np.zeros(len(meta), bool)
    for k, (tk, pp) in enumerate(zip(meta["ticker"], p)):
        j = col.get(tk)
        if j is None or pp < 1 or pp >= len(idx):
            continue
        dec = pp - 1                                          # 판단일 i 가 dec..dec+4 이면 거부에 걸림
        need_c[k] = cand[dec:min(dec + 5, len(idx)), j].any()
    samp = np.array([sampled(a) for a in meta["acc"]])
    return need_c, samp


def do_score():
    meta = pd.DataFrame(json.loads(x) for x in open(META))
    need_c, samp = needed(meta)
    meta["cand_window"], meta["sampled"] = need_c, samp
    todo = meta[need_c | samp]
    done = set()
    if os.path.exists(OUT):
        for x in open(OUT):
            try:
                done.add(json.loads(x)["acc"])
            except Exception:
                pass
    todo = todo[~todo["acc"].isin(done)]
    print(f"[v35] 목록 {len(meta)}건 중 받을 것 {int((need_c | samp).sum())}건 "
          f"(후보 창 {int(need_c.sum())}, 무작위 표본 {int(samp.sum())}) | 남은 {len(todo)}건", flush=True)
    lm = f.load_dict()
    lock = f.threading.Lock()
    t0, n = time.time(), 0

    def job(r):
        rec = f.one_filing(r["cik"], r["ticker"], {k: r[k] for k in
                                                   ("acc", "date", "accepted", "primary", "items")}, lm)
        rec["cand_window"], rec["sampled"] = bool(r["cand_window"]), bool(r["sampled"])
        return rec

    with open(OUT, "a") as fo, ThreadPoolExecutor(WORKERS) as ex:
        for rec in ex.map(job, todo.to_dict("records")):
            with lock:
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n += 1
                if n % 1000 == 0:
                    print(f"  {n}/{len(todo)}건, {(time.time() - t0) / 60:.0f}분", flush=True)
    print(f"[v35] 점수 완료 {n}건, {(time.time() - t0) / 60:.0f}분", flush=True)


if __name__ == "__main__":
    {"list": do_list, "score": do_score}[sys.argv[1]]()
