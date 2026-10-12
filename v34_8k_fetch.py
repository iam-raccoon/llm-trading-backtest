# -*- coding: utf-8 -*-
"""v34 수집기 — 미장 유니버스 587종목의 8-K 공시를 SEC EDGAR 에서 받아 **단어 사전 점수**만 저장.

사전 = Loughran-McDonald Master Dictionary (sraf.nd.edu, 2026-03판, 학술 연구용 무료).
원문은 저장하지 않는다. 공시마다 (총 단어, 부정어, 긍정어, 불확실, 소송) 개수와 접수 시각만 남긴다.

· 대상: form == "8-K" (정정 8-K/A 제외), 접수일 2021-06-01 ~ 2024-12-31
  (봉인 구간 2025~ 는 받지 않는다)
· 문서: 본문(primaryDocument) + 첨부 EX-99.* (실적 보도자료가 여기 들어간다), 최대 4개
· SEC 공정 접근 규칙: User-Agent 에 연락처, 초당 10회 미만 → 요청 간 0.13초 이상
· 이어 받기: 결과는 JSONL 에 한 줄씩 추가, 이미 있는 접수번호는 건너뜀

사용: nohup python3 -u v34_8k_fetch.py > cache_v34.log 2>&1 &
"""
import html
import json
import os
import pickle
import re
import time
import urllib.error
import urllib.request

import pandas as pd

UA = "llm-trading-backtest research kimdaehwankor@gmail.com"
OUT_DIR = "/home/user/llm-trading-backtest/cache/v34_8k"
OUT = os.path.join(OUT_DIR, "scores.jsonl")
DICT = os.path.join(OUT_DIR, "lm_dict.csv")
PX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "v26_us_px.pkl")
D0, D1 = "2021-06-01", "2024-12-31"
GAP = 0.13
MAX_DOC_BYTES = 3_000_000

_last = [0.0]


def get(url, tries=5):
    for k in range(tries):
        wait = GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read(MAX_DOC_BYTES)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(5 * (k + 1) if e.code in (429, 403, 503) else 2)
        except Exception:
            time.sleep(3 * (k + 1))
    return None


def load_dict():
    d = pd.read_csv(DICT)
    w = d["Word"].astype(str).str.upper()
    return {name: set(w[d[col] > 0]) for name, col in
            (("neg", "Negative"), ("pos", "Positive"), ("unc", "Uncertainty"), ("lit", "Litigious"))}


TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"[A-Za-z]+")


def score(raw, lm):
    t = raw.decode("utf-8", "ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = html.unescape(TAG.sub(" ", t))
    words = [x.upper() for x in WORD.findall(t) if len(x) > 1]
    c = {"words": len(words)}
    for k, s in lm.items():
        c[k] = sum(1 for x in words if x in s)
    return c


def ticker_ciks(tickers):
    j = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
    m = {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in j.values()}
    out = {}
    for t in tickers:
        k = t.upper().replace("-", ".")
        if k in m:
            out[t] = m[k]
    return out


def filings_8k(cik):
    base = "https://data.sec.gov/submissions/"
    j = json.loads(get(f"{base}CIK{cik:010d}.json") or b"{}")
    blocks = [j.get("filings", {}).get("recent", {})]
    for f in j.get("filings", {}).get("files", []):
        if f.get("filingTo", "9999") >= D0 and f.get("filingFrom", "0000") <= D1:
            jj = get(base + f["name"])
            if jj:
                blocks.append(json.loads(jj))
    rows = []
    for b in blocks:
        n = len(b.get("form", []))
        for i in range(n):
            if b["form"][i] != "8-K":
                continue
            fd = b["filingDate"][i]
            if not (D0 <= fd <= D1):
                continue
            rows.append({"acc": b["accessionNumber"][i], "date": fd,
                         "accepted": b.get("acceptanceDateTime", [""] * n)[i],
                         "primary": b.get("primaryDocument", [""] * n)[i],
                         "items": b.get("items", [""] * n)[i]})
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lm = load_dict()
    tickers = list(pickle.load(open(PX, "rb")).keys())
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["acc"])
            except Exception:
                pass
    cmap = ticker_ciks(tickers)
    print(f"[v34] 종목 {len(tickers)} 중 CIK 매칭 {len(cmap)} | 이미 받은 공시 {len(done)}", flush=True)
    t0, n_new = time.time(), 0
    with open(OUT, "a") as fo:
        for ti, (tk, cik) in enumerate(cmap.items(), 1):
            try:
                rows = filings_8k(cik)
            except Exception as e:
                print(f"  ⚠ {tk} 목록 실패 {e}", flush=True)
                continue
            for r in rows:
                if r["acc"] in done:
                    continue
                accn = r["acc"].replace("-", "")
                idx = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/index.json")
                names = []
                if idx:
                    items = json.loads(idx).get("directory", {}).get("item", [])
                    names = [x["name"] for x in items]
                docs = [r["primary"]] if r["primary"] else []
                docs += [n for n in names if re.search(r"ex[-_]?99", n, re.I)
                         and n.lower().endswith((".htm", ".html", ".txt")) and n not in docs]
                tot = {"words": 0, "neg": 0, "pos": 0, "unc": 0, "lit": 0}
                for dname in docs[:4]:
                    raw = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{dname}")
                    if raw:
                        for k, v in score(raw, lm).items():
                            tot[k] += v
                rec = {"ticker": tk, "cik": cik, **r, "ndocs": len(docs[:4]), **tot}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                done.add(r["acc"])
                n_new += 1
            if ti % 25 == 0:
                el = time.time() - t0
                print(f"  {ti}/{len(cmap)} 종목, 새 공시 {n_new}건, {el / 60:.0f}분", flush=True)
    print(f"[v34] 완료 — 새 공시 {n_new}건, {(time.time() - t0) / 60:.0f}분", flush=True)


if __name__ == "__main__":
    main()
