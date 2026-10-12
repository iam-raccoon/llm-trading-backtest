# -*- coding: utf-8 -*-
"""8-K 악재 그림자 기록 — **매매에는 쓰지 않는다.** v34 사후 결과를 앞으로의 데이터로 확인하는 장치.

━━━ 무엇을 하나 ━━━
실전 봇(us_trend_paper.py)이 매일 남기는 돌파 후보 목록(cache/us_cands_log.jsonl)을 읽고,
후보마다 '최근 5거래일 안에 접수된 8-K' 를 SEC EDGAR 에서 받아 Loughran-McDonald 사전 점수를 매긴다.
tone = (긍정어 − 부정어) ÷ 총 단어 ≤ THR (그리고 200단어 이상) 이면 '악재 표시'.
결과는 cache/us_news_shadow.jsonl 에 한 줄씩. 나중에 표시/비표시 후보의 20일 수익을 비교한다.

━━━ 백테스트와 같게 맞춘 것 (v34) ━━━
· 악재 창: 판단일(asof = 신호 종가 날짜) 포함 최근 5거래일. 공시의 '첫 거래 가능 시가'가
  (asof−4 거래일)의 다음 시가 ~ (asof 다음 거래일 = 봇 실행일) 시가 사이면 포함.
  = 접수 시각(미 동부)이 [asof−4 거래일 09:30, 봇 실행일 09:30) 구간.
· THR = −0.007868: 2024년 8-K 6,782건(200단어 이상)의 하위 20%. 백테스트의 '직전 252일 하위 20%'를
  실전에선 매일 전 종목 공시를 받지 않으므로 고정값으로 쓴다(전 기간 값 −0.007921 과 거의 같음).

━━━ 언제 도나 ━━━
한국 07:30(화~토). SEC 만 쓰고 토스는 안 쓴다 → 실전 봇 토큰과 충돌 없음.
SEC 공정 접근: UA 연락처(사용자 지정), 요청 간 0.15초 이상.

━━━ 판정 계획 (지금 고정) ━━━
표시 후보 n ≥ 300(종목-일) 이 쌓이면: 표시 − 비표시 20일 시장 대비 초과수익 차이 < 0 이고 t ≤ −2 →
실전 '악재 뒤 5일 신규 매수 거부' 도입을 사용자에게 제안. 그 전엔 6개월마다 방향만 본다.

사용: python3 us_news_shadow.py
"""
import datetime as dt
import html
import json
import os
import re
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CANDS = os.path.join(HERE, "cache", "us_cands_log.jsonl")
OUT = os.path.join(HERE, "cache", "us_news_shadow.jsonl")
DONE = os.path.join(HERE, "cache", "us_news_shadow_done.json")
WORDS = os.path.join(HERE, "cache", "lm_negpos.json")
UA = "llm-trading-backtest research kimdaehwankor@gmail.com"
THR = -0.007868
MIN_WORDS = 200
WINDOW = 5
GAP = 0.15
_last = [0.0]


def get(url, tries=4):
    for k in range(tries):
        w = GAP - (time.time() - _last[0])
        if w > 0:
            time.sleep(w)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read(3_000_000)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(5 * (k + 1))
        except Exception:
            time.sleep(3 * (k + 1))
    return None


TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"[A-Za-z]+")


def score(raw, neg, pos):
    t = raw.decode("utf-8", "ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    words = [x.upper() for x in WORD.findall(html.unescape(TAG.sub(" ", t))) if len(x) > 1]
    return len(words), sum(x in neg for x in words), sum(x in pos for x in words)


def sessions_back(d, n):
    """d 에서 n 거래일 앞(주말·NYSE 휴장 제외)."""
    from us_trend_paper import is_session
    k = 0
    while k < n:
        d -= dt.timedelta(days=1)
        if is_session(d):
            k += 1
    return d


def ny(ts_utc):
    import pandas as pd
    return pd.Timestamp(ts_utc).tz_convert("America/New_York")


def main():
    import pandas as pd
    lm = json.load(open(WORDS))
    neg, pos = set(lm["neg"]), set(lm["pos"])
    cmap = {v["ticker"].upper().replace("-", "."): int(v["cik_str"])
            for v in json.loads(get("https://www.sec.gov/files/company_tickers.json")).values()}
    done = set(json.load(open(DONE))) if os.path.exists(DONE) else set()
    lines = [json.loads(x) for x in open(CANDS)] if os.path.exists(CANDS) else []
    todo = [x for x in lines if x["us_date"] not in done]
    print(f"===== 8-K 그림자 기록 {dt.date.today()} | 새 실행일 {len(todo)}개 =====")
    subs = {}
    for day in todo:
        run_d = dt.date.fromisoformat(day["us_date"])
        asof = dt.date.fromisoformat(day["asof"])
        lo = pd.Timestamp(sessions_back(asof, WINDOW - 1), tz="America/New_York") \
            + pd.Timedelta(hours=9, minutes=30)
        hi = pd.Timestamp(run_d, tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        nflag = 0
        with open(OUT, "a", encoding="utf-8") as fo:
            for c in day["cands"]:
                sym = c["sym"]
                cik = cmap.get(sym.upper().replace("-", "."))
                rec = {"us_date": day["us_date"], "asof": day["asof"], "sym": sym,
                       "rank": c["rank"], "close": c["close"], "held": sym in day.get("held", []),
                       "filings": [], "flag": False, "cik": cik}
                if cik:
                    if cik not in subs:
                        j = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
                        subs[cik] = json.loads(j)["filings"]["recent"] if j else {}
                    r = subs[cik]
                    for i, form in enumerate(r.get("form", [])):
                        if form != "8-K":
                            continue
                        acc_t = ny(r["acceptanceDateTime"][i])
                        if not (lo <= acc_t < hi):
                            continue
                        accn = r["accessionNumber"][i].replace("-", "")
                        idx = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/index.json")
                        names = [x["name"] for x in json.loads(idx)["directory"]["item"]] if idx else []
                        docs = [r["primaryDocument"][i]] + [
                            n for n in names if re.search(r"ex[-_]?99", n, re.I)
                            and n.lower().endswith((".htm", ".html", ".txt"))
                            and n != r["primaryDocument"][i]]
                        w = ng = ps = 0
                        for dn in docs[:4]:
                            raw = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{dn}")
                            if raw:
                                a, b, cc = score(raw, neg, pos)
                                w, ng, ps = w + a, ng + b, ps + cc
                        tone = (ps - ng) / w if w else None
                        bad = bool(w >= MIN_WORDS and tone is not None and tone <= THR)
                        rec["filings"].append({"acc": r["accessionNumber"][i],
                                               "accepted": str(acc_t), "items": r["items"][i],
                                               "words": w, "tone": tone, "bad": bad})
                        rec["flag"] |= bad
                nflag += rec["flag"]
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done.add(day["us_date"])
        json.dump(sorted(done), open(DONE, "w"))
        print(f"  {day['us_date']} (신호 {day['asof']}): 후보 {len(day['cands'])}개 중 악재 표시 {nflag}개")
    print("  끝 — 매매에는 쓰지 않는 기록이다")


if __name__ == "__main__":
    main()
