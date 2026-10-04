# -*- coding: utf-8 -*-
"""미장 추세추종 — 포워드 페이퍼 / 실행. **기본은 페이퍼, 실주문은 이중 잠금.**

━━━ 규칙 (v26 선정, 고정) ━━━
    유니버스  S&P500 + 나스닥 상위 587종목
    진입      직전 **40봉** 고가 돌파  그리고  종가 > **MA120**
              후보는 **20일 모멘텀** 순, 슬롯 **3개**
    청산      샹들리에 트레일링 — 보유 중 최고가 − **3.0 × ATR14**
    국면필터  **없음**(regime0). 한국과 다른 점 — 미장은 IS 탐색에서 필터가
              선택되지 않았고, 하락장이었던 2022 조각에서도 Sharpe 0.82 로 버텼다.
    비용      왕복 0.20% (수수료 0.1%×2, 거래세 없음) + 환전 스프레드(미반영)

**소수점 매수를 쓴다.** 토스 API 명세상 `orderAmount`(금액 지정)는 US MARKET 전용이다.
$72.5 를 3등분하면 슬롯당 약 $24.17 이고, 1주 단위 제약이 없어 비중이 정확히 맞는다.
(한국은 1주 단위라 10만원/3슬롯이면 유니버스의 49%만 살 수 있다.)

━━━ 검증 상태 ━━━
IS 2021-09~2024-12 에서 3조각 maximin 으로 선정 → OOS 2025-01~2026-08 을 **한 번** 열었다.
    IS  +177.1%  SR 1.22  |  OOS +78.9%  SR 1.21  |  대조군 +41.5%
IS·OOS Sharpe 가 거의 같아 과최적화 감쇠가 없다. v27 에서 개선(리스크 패리티·섹터제한)을
시도했으나 IS 에서 선택되지 않아 **원본을 그대로 쓴다.**

━━━ 안전장치 (toss_trade 와 동일 철학) ━━━
`--live` 플래그 **그리고** 환경변수 `TOSS_LIVE=1` 이 둘 다 있어야 실주문이 나간다.
하나만으로는 안 나간다. 주문 금액은 슬롯 크기로 제한되고, 계좌 잔고를 넘지 않는다.

━━━ 사전 기준 (기록 쌓기 전에 고정) ━━━
6개월 또는 거래 30건 중 나중에 오는 시점에:
  (a) 거래당 평균 > 0.20%  (b) 누적 > 유니버스 동일가중  (c) MDD > −30%  (d) 거래 30건+

사용: python3 us_trend_paper.py              # 페이퍼(기본)
      python3 us_trend_paper.py --signals    # 오늘 신호만 보기
      TOSS_LIVE=1 python3 us_trend_paper.py --live   # 실주문(승인 후)
"""
import argparse
import datetime as dt
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "cache", "us_trend_state.json")
PXC = os.path.join(HERE, "cache", "v26_us_px.pkl")

# ★ 2026-09-26: 고정값을 버리고 **분기 재선택 설정파일**을 읽는다.
# 이유(v28/v29 워크포워드):
#   · IS 성적 +177% 는 '그 구간에서 고른 이득'이었고 공정하게 재면 +45.9%(SR 0.64).
#   · 창마다 최적 설정이 계속 바뀐다 — **고정된 최적값이 없다.**
#   · 슬롯 3 은 IS·봉인 두 구간 모두 꼴찌였고 5·10 이 크게 나았다
#     (봉인: 슬롯3 +74.0/SR1.07, 슬롯10 +150.3/SR1.86, 시장 +41.5/SR1.35).
# 검증한 방식이 '분기 재선택'이므로 그대로 굴린다. 파라미터만 고정하면
# 검증한 것과 다른 물건이 된다(v22 의 실패와 같은 함정).
CONFIG = os.path.join(HERE, "cache", "us_trend_config.json")
_DEFAULT = {"dc": 40, "ma": 120, "chand": 3.0, "mom": 20, "slots": 3}


def load_config():
    """분기 재선택 설정. 없거나 묵었으면 경고하되 멈추지는 않는다."""
    cfg = dict(_DEFAULT)
    try:
        j = json.load(open(CONFIG, encoding="utf-8"))
        cfg.update({k: j[k] for k in ("dc", "ma", "chand", "mom", "slots") if k in j})
        cfg["chosen_at"] = j.get("chosen_at")
        nr = j.get("next_review")
        if nr and dt.date.fromisoformat(nr) < dt.date.today():
            print(f"  ⚠️ 파라미터 재선택 예정일({nr})이 지났다 — us_reselect.py 실행 요망")
    except Exception:
        print(f"  ⚠️ 설정파일 없음 — 기본값 사용(슬롯 {cfg['slots']}). "
              f"us_reselect.py 를 먼저 돌릴 것")
    return cfg


_C = load_config()
DC_ENTRY, MA_TREND = _C["dc"], _C["ma"]
CHAND, MOM, SLOTS = _C["chand"], _C["mom"], _C["slots"]
ATR_N = 14
COST = 0.20
CAP_USD = 72.5
DEADLINE = 240
# 실측 비용 구조에서 온 하한(us_costs.py 참조): 수수료는 센트 미만 절사라 $10 미만
# 주문은 0원이지만, **매도 세금(SEC fee)은 최소 $0.01 고정**이다. 그래서 슬롯이
# 작아질수록 비용률이 커진다 — $7 이면 왕복 0.14%, $3 이면 0.33%, $2 면 0.50%.
# 백테스트 가정(0.20%)을 넘지 않는 선이 슬롯 $5 다.
MIN_ORDER_USD = 5.0


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE, encoding="utf-8"))
        except Exception:
            pass
    return {"start": dt.date.today().isoformat(), "cash": CAP_USD,
            "pos": {}, "closed": [], "equity": []}


def save_state(s):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(s, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def symbols():
    """캐시된 유니버스. 없으면 받아서 캐시한다."""
    if os.path.exists(PXC):
        raw = pickle.load(open(PXC, "rb"))
        return [(c, v["name"]) for c, v in raw.items()]
    raise RuntimeError("유니버스 캐시가 없다 — v26 데이터를 먼저 받을 것")


# NYSE 휴장일·조기마감(13:00 ET). 출처: NYSE Group 2025~2027 공식 발표.
# ⚠️ 2027 말에 2028 분을 추가할 것 — 없으면 휴장일에 주문이 나가 거부된다.
NYSE_HOLIDAYS = {dt.date.fromisoformat(d) for d in (
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24")}
NYSE_EARLY_CLOSE = {dt.date.fromisoformat(d) for d in (
    "2026-11-27", "2026-12-24", "2027-11-26")}


def ny_now():
    """미국 동부 현재 시각. **서머타임을 자동 반영한다.**

    ⚠️ 전에는 UTC−4 고정이었다. 2026-11-01 서머타임이 끝나면 한국 22:40 은
    뉴욕 08:40(개장 전)인데 코드는 09:40 이라고 믿었을 것이다."""
    return pd.Timestamp.now(tz="America/New_York")


def us_today():
    """미국 동부 기준 '오늘' 날짜.

    한국 밤 22:40 은 미국 당일 개장 직후(서머타임 기준)다. 즉 **장중에 실행된다.**
    이때 데이터 소스가 오늘의 **미완성 봉**을 마지막 행으로 줄 수 있는데,
    그걸 종가로 쓰면 신호가 백테스트와 달라진다(백테스트는 완료된 종가로 신호를
    만들고 다음 시가에 산다). 그래서 오늘 날짜 행은 잘라낸다."""
    return ny_now().date()


def is_session(d):
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def market_window():
    """지금 주문을 내도 되는가 → (bool, 사유).

    정규장 개장 5분 뒤부터 **마감 1시간 전**까지만 허용한다. 소수점 시장가 매도는
    마감 1시간 전까지만 되고, 금액 주문은 정규장 전용이다. 휴장일·장외에 돌면
    주문이 거부되어 프로그램이 죽거나(9/30 의 422 처럼), 더 나쁘게는 프리마켓
    같은 얇은 호가에 시장가가 나갈 수 있다."""
    now = ny_now()
    d = now.date()
    if not is_session(d):
        return False, f"뉴욕 {d} 휴장일"
    t = now.time()
    close = dt.time(12, 0) if d in NYSE_EARLY_CLOSE else dt.time(15, 0)
    if not (dt.time(9, 35) <= t <= close):
        return False, f"뉴욕 {now:%H:%M} — 주문 가능 시간(09:35~{close:%H:%M}) 밖"
    return True, f"뉴욕 {now:%H:%M}"


def indicators(d):
    d = d.dropna(subset=["Close"])
    # ⚠️ 장중 실행 시 오늘의 미완성 봉이 붙는다. 신호는 **직전 완료 세션**으로만.
    cut = us_today()
    d = d[d.index.date < cut]
    if len(d) < MA_TREND + 30:
        return None
    hi, lo, cl = d["High"], d["Low"], d["Close"]
    pc = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return {"close": float(cl.iloc[-1]), "high": float(hi.iloc[-1]),
            "atr": float(tr.rolling(ATR_N).mean().iloc[-1]),
            "dc": float(hi.rolling(DC_ENTRY).max().shift(1).iloc[-1]),
            "ma": float(cl.rolling(MA_TREND).mean().iloc[-1]),
            "mom": float(cl.iloc[-1] / cl.iloc[-1 - MOM] - 1),
            # ★ 이 봉이 **언제 것인지**. 신선도 검사에 쓴다.
            "asof": cl.index[-1].date()}


def expected_session(today=None):
    """직전 **완료** 미국 거래일(주말·NYSE 휴장일 건너뜀).

    장중(09:30~16:00 ET)에 실행하므로 오늘 봉은 미완성이고, 신호는 어제(직전
    영업일) 종가로 만들어야 한다. 휴장일을 몰랐을 때는 휴장 다음 날 모든 보유
    종목이 '데이터 묵음'으로 판정 보류되어 손절을 하루 놓쳤다."""
    d = (today or us_today()) - dt.timedelta(days=1)
    while not is_session(d):
        d -= dt.timedelta(days=1)
    return d


def check_freshness(ind):
    """데이터가 **예상 직전 거래일** 것인지 확인한다.

    ⚠️ 2026-08-31 첫 실주문이 **8/27 종가**로 신호를 만들었다. 예상은 8/28 이었다.
    데이터 소스가 최신 세션을 아직 안 올렸는데 코드가 마지막 행을 그냥 썼기 때문이다.
    그사이 MSTR 은 8/27 $137.40 → 8/28 $127.31 로 −7.3% 빠졌는데, 시스템은
    그걸 못 보고 "모멘텀 +40.6%" 로 매수했다. **떨어지는 걸 모르고 산 것이다.**

    묵은 신호로 매매하는 것은 백테스트와 다른 전략을 굴리는 것이므로,
    하루 거르는 손해보다 크다. 어긋나면 **중단한다.**
    공휴일이면 오탐이 나는데, 그때는 하루 쉬는 것이 맞다
    (`ALLOW_STALE=1` 로 강제 진행은 가능하게 두되 기본은 중단)."""
    from collections import Counter
    dates = Counter(x["asof"] for _, x in ind.values())
    if not dates:
        raise RuntimeError("신선도 확인 불가 — 데이터가 없다")
    mode, n = dates.most_common(1)[0]
    exp = expected_session()
    share = 100 * n / len(ind)
    print(f"  데이터 기준일 {mode} ({share:.0f}% 종목 일치) | 예상 {exp}")
    if len(dates) > 1:
        others = ", ".join(f"{d}:{c}" for d, c in dates.most_common()[1:4])
        print(f"    · 다른 날짜 섞임 — {others}")
    if mode == exp:
        return mode
    behind = (exp - mode).days
    msg = (f"데이터가 예상보다 {behind}일 묵었다 (기준 {mode} / 예상 {exp}). "
           f"묵은 신호로 매매하면 백테스트와 다른 전략이 된다.")
    if os.environ.get("ALLOW_STALE") == "1":
        print(f"  ⚠️ {msg}  → ALLOW_STALE=1 이라 강제 진행")
        return mode
    raise RuntimeError(msg + " 오늘은 매매하지 않는다 "
                       "(공휴일이면 정상 동작이다. 강제하려면 ALLOW_STALE=1)")


def toss_candles(t, sym, count=200):
    """토스 API 일봉 → DataFrame(Open/High/Low/Close), 날짜 인덱스.

    ★ FinanceDataReader 를 버리고 여기로 옮긴 이유:
    **FDR 의 미국 일봉이 한 세션씩 늦다.** 2026-09-04 09:34 KST 기준으로 FDR 의
    마지막 봉은 9/2 였고 9/3 이 통째로 없었다. 그날 MSTR 은 $123.19 → $144.87 로
    **+17.6% 튀었는데** 그걸 못 보고 있었다. 8/31 첫 실주문이 8/27 데이터로 나간 것도
    같은 원인이다.

    증권사 API 에 캔들이 있는데 늦는 제3자 소스를 쓸 이유가 없다. 게다가
    **주문을 내는 곳과 시세를 보는 곳이 같아야** 체결가와 신호가 어긋나지 않는다.

    ⚠️ 최대 200봉. MA120 + 여유를 쓰므로 충분하지만 더 긴 지표는 못 쓴다."""
    r = t._call("GET", "/api/v1/candles",
                {"symbol": sym, "market": "US", "interval": "1d",
                 "count": count, "adjusted": "true"})
    rows = ((r or {}).get("result") or {}).get("candles") or []
    if not rows:
        return None
    recs = []
    for c in rows:
        recs.append({
            "date": pd.Timestamp(c["timestamp"]).date(),
            "Open": float(c["openPrice"]), "High": float(c["highPrice"]),
            "Low": float(c["lowPrice"]), "Close": float(c["closePrice"])})
    d = pd.DataFrame(recs).set_index("date").sort_index()
    d.index = pd.to_datetime(d.index)
    return d


def fetch_all(syms, t=None):
    """⚠️ `t` 를 반드시 넘길 것. 여기서 Toss() 를 새로 만들면 **새 토큰이 발급되어
    호출부의 토큰이 죽는다**(토스는 신규 발급 시 이전 토큰 무효화). 실제로
    2026-09-29 에 매도는 됐는데 매수가 401 token-revoked 로 거부되고 현금이
    3일간 놀았다."""
    if t is None:
        from toss_trade import Toss
        t = Toss()

    def one(it):
        s, n = it
        try:
            d = toss_candles(t, s)
            if d is None:
                return None
            x = indicators(d)
            return (s, n, x) if x else None
        except Exception:
            return None

    out, done, t0 = {}, 0, time.time()
    # MARKET_DATA 는 초당 15 제한이라 워커를 낮춘다(429 방지)
    ex = ThreadPoolExecutor(max_workers=6)
    futs = {ex.submit(one, it): it for it in syms}
    try:
        for fu in as_completed(futs, timeout=DEADLINE):
            done += 1
            if done % 150 == 0:
                print(f"    {done}/{len(syms)}  {time.time()-t0:.0f}초", flush=True)
            try:
                r = fu.result(timeout=1)
            except Exception:
                continue
            if r:
                out[r[0]] = (r[1], r[2])
    except TimeoutError:
        print(f"  ⚠️ 마감 {DEADLINE}초 — {done}/{len(syms)} 까지만", flush=True)
    for fu in futs:
        fu.cancel()
    ex.shutdown(wait=False)
    return out


def fetch_one(t, sym, since=None, tries=4):
    """보유 종목용 — **반드시 받아낸다.** DNS 가 간헐적으로 튕기므로 재시도한다.

    `since`(진입일)를 주면 그 이후 완료 세션들의 **최고가(peak_since)** 도 돌려준다.
    ⚠️ 최고가를 '실행한 날의 고가'로만 누적했더니, 봇이 조회 실패로 건너뛴 날의
    고가를 놓쳐 MSTR 손절선이 $133.78 이어야 할 것이 $124.82 로 덜 올라왔다.
    캔들에서 매번 다시 계산하면 건너뛴 날이 있어도 손절선이 정확하다."""
    import time as _t
    for k in range(tries):
        try:
            d = toss_candles(t, sym)
            if d is not None:
                x = indicators(d)
                if x:
                    if since:
                        dd = d[(d.index.date >= dt.date.fromisoformat(since))
                               & (d.index.date < us_today())]
                        x["peak_since"] = float(dd["High"].max()) if len(dd) else None
                        # 캔들 200봉이 진입일까지 거슬러 올라가는가
                        x["peak_covers"] = bool(
                            len(d) and d.index[0].date() <= dt.date.fromisoformat(since))
                    return x
        except Exception:
            pass
        _t.sleep(2 * (k + 1))
    return None


def account_positions(t):
    """실계좌 보유(심볼 → 수량, 평단, 현재가). 매도 수량은 **이것**을 쓴다."""
    h = t.holdings(market="US") or {}
    items = []
    for v in (h.values() if isinstance(h, dict) else []):
        if isinstance(v, list):
            items = v
    return {x["symbol"]: {"qty": float(x["quantity"]),
                          "avg": float(x["averagePurchasePrice"]),
                          "last": float(x["lastPrice"])} for x in items}


def main(a):
    """하루 1회. 순서가 중요하다 — **① 보유 종목 청산 판정 → ② 신규 진입.**

    ⚠️ 2026-09-21 에 고친 것 셋 (전부 청산이 한 번도 안 나서 안 드러났다):
      1. **실주문 모드에서 매도 주문을 아예 안 냈다.** 장부에서만 지우고 실제 주식은
         그대로 뒀다. 손절선에 걸려도 **실제로는 절대 안 팔리는 구조**였다.
      2. 유니버스 조회가 70% 미만이면 **통째로 중단**했다. 7일 중 4일이 그랬고
         그날은 **손절 판정도 안 했다.** 급락했으면 못 팔았다.
         → 이제 조회 부족은 **신규 진입만** 막는다. 청산 판정은 보유 종목을
           따로(재시도 포함) 받아서 반드시 한다.
      3. 조회가 반쯤 된 날 보유 종목이 빠져 평가액을 −70% 로 기록했다(실제 +7.65%).
         → 실주문 모드에선 **실계좌 잔고**로 평가액을 기록한다.
    """
    # ★ 페이퍼 실행은 **별도 장부**에 쓴다. 전에는 `--live` 를 빼고 돌리면 실계좌
    # 장부(us_trend_state.json)에 가짜 매매를 기록해 버렸다.
    global STATE
    if not a.signals and not (a.live and os.environ.get("TOSS_LIVE") == "1"):
        STATE = STATE.replace(".json", "_paper.json")
    st = load_state()
    # ★ '오늘' = **뉴욕 날짜**. 한국 날짜를 쓰면 자정 넘은 예비 실행(겨울 00:40)이
    # 같은 미국 세션을 '새 날'로 보고 한 번 더 매매한다.
    today = us_today().isoformat()
    print(f"===== 미장 추세추종 {'[실주문]' if a.live else '[페이퍼]'}  {today} "
          f"(시작 {st['start']}) =====")
    print(f"  규칙: DC{DC_ENTRY} MA{MA_TREND} 샹들리에{CHAND}xATR 모멘텀{MOM} "
          f"슬롯{SLOTS} 국면필터없음 (선택일 {_C.get('chosen_at', '기본값')})")
    print(f"  신호 기준: 미국 {us_today()} **이전** 완료 세션")

    # ── 하루 한 번 보장 ──
    # 22:40 본 실행 + 23:40 예비 실행을 둔다(DNS 장애로 7일 중 4일 실패한 적이 있다).
    # 예비 실행은 **그날 본 실행이 이미 성공했으면 아무것도 안 한다.** 판단은 하루 한 번,
    # 실패한 날만 메운다. 수동으로 다시 돌려도 같은 날 두 번 매매하지 않는다.
    if not a.signals and st.get("last_ok") == today and not a.force:
        print(f"  오늘({today}) 이미 성공적으로 실행됨 — 종료 (강제하려면 --force)")
        return

    from toss_trade import Toss
    t = Toss()
    live = a.live and not a.signals
    if live and os.environ.get("TOSS_LIVE") != "1":
        print("  ⚠️ --live 인데 TOSS_LIVE!=1 — 페이퍼로만 진행")
        live = False
    if live:
        ok, why = market_window()
        print(f"  장 시간 확인: {why}")
        if not ok:
            # last_ok 를 남기지 않는다 → 장중의 다음 예약 실행이 처리한다
            print("  주문 불가 시간 — 아무것도 안 하고 종료")
            return
    # 주문이 하나라도 에러로 끝나면 last_ok 를 안 남겨 예비 실행이 다시 시도하게 한다.
    # (체결분은 즉시 저장되고 실계좌와 대조하므로 다시 돌아도 중복 매매하지 않는다)
    had_error = False

    # ── ① 보유 종목 청산 판정 (유니버스 조회와 무관하게 반드시) ──
    acct = account_positions(t) if live else {}
    exp = expected_session()
    for sym in list(st["pos"]):
        p = st["pos"][sym]
        # ★ 오늘 산 종목은 판정하지 않는다 — 진입 뒤 완료된 봉이 아직 없다.
        # 9/30 첫 실행이 MRNA 를 사고 죽은 뒤 예비 실행이 이걸 판정하면서
        # **사기 전날(9/29) 고가 $208.90** 을 최고가로 넣었다. 손절선이
        # $173.54 여야 할 것이 $181.75 로 $8 높게 박혀 일찍 털릴 뻔했다.
        # (백테스트도 진입 봉이 **끝난 뒤**부터 최고가·손절을 잰다.)
        if p.get("date") == today:
            print(f"  · {sym:<6} 오늘 진입 — 판정은 내일부터")
            continue
        x = fetch_one(t, sym, since=p.get("date"))
        if x is None:
            print(f"  ⚠️ {sym} 시세 조회 실패(재시도 4회) — 오늘 청산 판정 불가")
            continue
        if x["asof"] != exp and os.environ.get("ALLOW_STALE") != "1":
            print(f"  ⚠️ {sym} 데이터 {x['asof']} ≠ 예상 {exp} — 판정 보류")
            continue
        # ★ 최고가는 **수정주가 캔들에서 매번 다시** 잰다. 저장된 peak 와 max 를
        # 취하면, 액면분할 뒤 캔들은 분할 반영가인데 저장값은 옛 가격이라
        # 손절선이 현재가 위로 올라가 **멀쩡한 종목을 시장가로 던진다.**
        # 캔들이 진입일까지 못 거슬러 갈 때(보유 ~190일 초과)만 저장값을 섞는다.
        if x.get("peak_since") and x.get("peak_covers"):
            p["peak"] = x["peak_since"]
        else:
            p["peak"] = max(p["peak"], x["high"], x.get("peak_since") or 0)
        stop = p["peak"] - CHAND * x["atr"]
        p["stop"] = round(stop, 4)
        print(f"  · {sym:<6} 종가 ${x['close']:,.2f}  손절선 ${stop:,.2f}  "
              f"여유 {(x['close']/stop-1)*100:+.1f}%")
        if x["close"] > stop:
            continue
        # 손절선 이탈 → 매도
        if a.signals:
            print(f"  ▣ [신호만] {sym} 청산 대상")
            continue
        if live:
            q = acct.get(sym, {}).get("qty")
            if not q:
                print(f"  ⚠️ {sym} 실계좌에 없음 — 장부만 정리")
                exit_px, got = x["close"], 0.0
            else:
                # 미장 시장가 매도는 소수점 수량 허용(정규장 마감 1시간 전까지)
                # ⚠️ 예외를 여기서 잡는다. 안 잡으면 한 종목 매도 거부가 **나머지
                # 종목의 손절 판정까지** 통째로 날린다.
                try:
                    r = t.order(sym, qty=q, side="SELL", market="US", confirm=True)
                    oid = ((r or {}).get("result") or {}).get("orderId")
                    fill = t.wait_fill(oid) if oid else None
                except Exception as e:
                    print(f"  ⚠️ {sym} 매도 주문 에러: {e} — 다음 실행 때 재시도")
                    had_error = True
                    continue
                if fill and fill.get("qty"):
                    amt = fill.get("amount")
                    exit_px = (amt / fill["qty"]) if amt else x["close"]
                    got = fill["qty"] * exit_px
                    print(f"  ▣ [실매도] {sym} {q:.6f}주 @${exit_px:,.2f}")
                else:
                    print(f"  ⚠️ {sym} 매도 체결 확인 실패({(fill or {}).get('status')}) "
                          f"— 장부 유지, 다음 실행 때 재시도")
                    had_error = True
                    continue
        else:
            exit_px = x["close"]
            got = p["qty"] * exit_px
            print(f"  ▣ [페이퍼] 청산 {sym} @${exit_px:,.2f}")
        st["cash"] += got * (1 - COST / 100 / 2) if not live else got
        ret = (exit_px / p["entry"] - 1) * 100 - COST
        st["closed"].append({"sym": sym, "name": p.get("name", ""), "ret": round(ret, 2),
                             "in": p["date"], "out": today,
                             "tainted": bool(p.get("tainted"))})
        print(f"      수익 {ret:+.2f}%{'  (오염 표시 거래)' if p.get('tainted') else ''}")
        del st["pos"][sym]
        save_state(st)      # ★ 주문 하나마다 즉시 저장 — 도중에 죽어도 중복 매매 방지

    # ── ② 신규 진입 (슬롯이 비었을 때만 유니버스를 받는다) ──
    held_now = set(st["pos"])
    if live:
        # 장부만 믿지 않는다. 산 직후 저장 전에 죽은 경우 실계좌엔 있고 장부엔 없다 —
        # 그 상태로 예비 실행이 돌면 **같은 종목을 또 사거나 슬롯을 초과**한다.
        acct = account_positions(t)
        held_now |= set(acct)
        st["cash"] = float(t.buying_power("USD")["cashBuyingPower"])
        orphan = set(acct) - set(st["pos"])
        if orphan:
            print(f"  ⚠️ 실계좌엔 있고 장부엔 없는 종목 {sorted(orphan)} — 슬롯으로 세고 "
                  f"신규 매수에서 제외(수동 확인 필요)")
    free = SLOTS - len(held_now)
    if free > 0 or a.signals:
        syms = symbols()
        print(f"\n  빈 슬롯 {free} — 유니버스 {len(syms)}종목 조회...", flush=True)
        ind = fetch_all(syms, t)      # ★ 같은 객체 재사용(토큰 충돌 방지)
        print(f"  조회 {len(ind)}/{len(syms)}종목")
        ok_uni = len(ind) >= len(syms) * 0.7
        if not ok_uni:
            print("  ⚠️ 조회 부족 — 신호 왜곡 우려로 **신규 진입만** 건너뛴다(청산은 위에서 끝남)")
        else:
            try:
                check_freshness(ind)
            except RuntimeError as e:
                print(f"  ⚠️ {e}")
                ok_uni = False
        if ok_uni:
            cands = sorted(((x["mom"], s_, nm, x["close"])
                            for s_, (nm, x) in ind.items()
                            if x["close"] > x["dc"] and x["close"] > x["ma"]
                            and not np.isnan(x["mom"])), reverse=True)
            print(f"  돌파 후보 {len(cands)}개")
            picks = [(s_, nm, px, m) for m, s_, nm, px in cands
                     if s_ not in held_now][:max(free, 0)]
            # ★ 슬롯 금액 = **전체 자산 ÷ 슬롯 수**.
            # 전에는 `현금 ÷ 빈슬롯` 이었다. 그러면 현금이 적을 때 터무니없이 작은
            # 주문이 나간다 — 2026-09-30 기준 현금 $21.87 / 빈슬롯 8 = **$2.73**.
            # 게다가 1기 보유분은 슬롯 3 시절에 $24 씩 산 것이라 새 기준($7.5)보다
            # 훨씬 크다. 그래서 **현금이 닿는 만큼만 사고**, 나머지 슬롯은 1기 종목이
            # 정리되면서 자연히 채워지게 둔다(억지로 크기를 줄이지 않는다).
            equity_now = st["cash"] + sum(
                p2["qty"] * (ind[k][1]["close"] if k in ind else p2["entry"])
                for k, p2 in st["pos"].items())
            slot = equity_now / SLOTS
            print(f"  슬롯 금액 ${slot:.2f} (자산 ${equity_now:.2f} ÷ {SLOTS}) "
                  f"| 현금 ${st['cash']:.2f} → 최대 {int(st['cash'] // slot)}종목")
            for s_, nm, px, m in (picks if not a.signals else []):
                if live:
                    # ★ 매수 직전마다 **실제 주문가능금액**을 다시 읽는다. 장부 차감
                    # (`cash -= amt`)만 믿었더니 9/30 세 번째 매수가 422 로 거부됐다
                    # (같은 금액이 1시간 뒤 예비 실행에선 체결 — 직전 체결분 정산 지연 추정).
                    try:
                        st["cash"] = float(t.buying_power("USD")["cashBuyingPower"])
                    except Exception as e:
                        print(f"  ⚠️ 주문가능금액 조회 실패: {e} — 매수 중단")
                        had_error = True
                        break
                amt = min(slot, st["cash"])
                if amt < MIN_ORDER_USD:
                    print(f"  · 현금 부족(${st['cash']:.2f}) — 남은 슬롯은 "
                          f"기존 종목 정리 후 채운다")
                    break
                if live:
                    try:
                        r = t.order(s_, side="BUY", market="US", amount=amt,
                                    confirm=True)
                        oid = ((r or {}).get("result") or {}).get("orderId")
                        fill = t.wait_fill(oid) if oid else None
                    except Exception as e:
                        # 거부된 주문이라 돈은 안 나갔다. 다음 후보로 넘어간다.
                        print(f"  ⚠️ {s_} 매수 주문 에러: {e}")
                        had_error = True
                        continue
                    if not (fill and fill.get("qty") and fill.get("fill_price")):
                        print(f"  ⚠️ {s_} 매수 체결 확인 실패 — 기록 안 함")
                        had_error = True
                        continue
                    qty, entry = fill["qty"], fill["fill_price"]
                    print(f"  ▶ [실매수] {s_} ${amt:.2f} → {qty:.6f}주 @${entry:,.2f} "
                          f"(신호가 대비 {(entry/px-1)*100:+.2f}%)")
                else:
                    qty, entry = amt / px, px
                    print(f"  ▶ [페이퍼] {s_} ${amt:.2f} @${px:,.2f}")
                st["cash"] -= amt
                free -= 1
                st["pos"][s_] = {"qty": qty, "entry": entry, "peak": entry,
                                 "date": today, "name": nm, "signal_px": px}
                save_state(st)      # ★ 즉시 저장
            if a.signals:
                for m, s_, nm, px in cands[:8]:
                    print(f"    {s_:<6}{nm[:22]:<24}${px:>9,.2f}  {m*100:+.1f}%")
    else:
        print(f"\n  슬롯 {SLOTS}/{SLOTS} 가득 — 신규 진입 없음(유니버스 조회 생략)")

    if a.signals:
        return

    # ── ③ 평가액: 실주문이면 **실계좌**로 ──
    if live:
        acct = account_positions(t)
        cash = float(t.buying_power("USD")["cashBuyingPower"])
        mv = cash + sum(v["qty"] * v["last"] for v in acct.values())
        st["cash"] = cash
        src = "실계좌"
    else:
        mv = st["cash"] + sum(p["qty"] * p["entry"] for p in st["pos"].values())
        src = "장부"
    print(f"\n  평가액({src}) ${mv:,.2f} ({(mv/CAP_USD-1)*100:+.2f}%)")
    clean = [c["ret"] for c in st["closed"] if not c.get("tainted")]
    if st["closed"]:
        print(f"  청산 {len(st['closed'])}건 (판정용·비오염 {len(clean)}건)")
    if st["equity"] and st["equity"][-1][0] == today:
        st["equity"][-1] = [today, round(mv, 4)]      # 같은 날 재실행은 덮어쓴다
    else:
        st["equity"].append([today, round(mv, 4)])
    if had_error:
        save_state(st)
        print(f"  ⚠️ 주문 에러가 있었다 — last_ok 를 남기지 않음(예비 실행이 재시도)")
        return
    st["last_ok"] = today           # ★ 여기까지 와야 '오늘 성공' — 예비 실행이 이걸 본다
    save_state(st)
    print(f"  {'실주문 모드' if live else '페이퍼 모드'} 종료 · last_ok={today}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--signals", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="오늘 이미 성공했어도 다시 실행(수동 전용)")
    main(p.parse_args())
