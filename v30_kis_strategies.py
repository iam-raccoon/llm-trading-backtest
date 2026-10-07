# -*- coding: utf-8 -*-
"""v30 — 한국투자증권 공식 레포의 전략 프리셋 10종을 우리 미장 데이터로 검증.

출처: github.com/koreainvestment/open-trading-api  strategy_builder/strategy/
규칙·기본 파라미터는 **원본 그대로** 옮겼다(우리가 고르지 않았으므로 이 구간에서
과최적화될 여지가 없다). 신호는 종가로 만들고 다음 날 시가에 체결한다.

━━━ 공정 비교 조건 ━━━
· 유니버스·슬롯 10·비용 0.20%·체결 규칙은 v28 과 같다(v28.sim 회계를 그대로 따름).
· 구간은 v28 워크포워드의 **검증 창들을 이어 붙인 기간**과 같다. 그래서 우리 봇의
  워크포워드 성적과 같은 날짜로 비교된다.
· 봉인 구간(2025-01~)은 쓰지 않는다 — 이미 두 번 열었다.
· 매수만 있는 전략(③⑦⑧)은 원본이 "손절 등 다른 전략과 조합"하라고 하므로
  우리 샹들리에(3×ATR)를 청산으로 붙인다. ⑥(매도 전용)은 우리 봇 청산에 덧붙인다.
· 매수 후보가 슬롯보다 많으면 원본의 `strength` 순으로 고른다. 동점이면 종목 순서
  (원본 프레임워크도 별도 순위 규칙이 없다).

사용: python3 v30_kis_strategies.py
"""
import pickle

import numpy as np
import pandas as pd

import v28_walkforward as v28
from v26_us_holdout import PX, build

SLOTS = 10
COST = v28.COST


KR_PX = "/home/user/llm-trading-backtest/cache/v24_px.pkl"   # 국장 413종목(v24)


def load_low(ind, px=PX):
    raw = pickle.load(open(px, "rb"))
    return pd.DataFrame({c: raw[c]["df"]["Low"] for c in ind["codes"]}).sort_index()


def build_kr():
    """국장을 미장과 같은 모양(ind)으로. 미장에서 고른 게 우연인지 가르는 **독립 시장 검사**."""
    raw = pickle.load(open(KR_PX, "rb"))
    codes = [c for c, v in raw.items() if len(v["df"]) > 300]
    f = lambda k: pd.DataFrame({c: raw[c]["df"][k] for c in codes}).sort_index()
    op, hi, lo, cl = f("Open"), f("High"), f("Low"), f("Close")
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo).stack(), (hi - pc).abs().stack(),
                    (lo - pc).abs().stack()], axis=1).max(axis=1).unstack()
    ind = {"op": op, "hi": hi, "cl": cl, "atr": tr.rolling(14).mean(), "codes": codes}
    ind["dc"] = {d: hi.rolling(d).max().shift(1) for d in v28.GRID["dc"]}
    ind["ma"] = {m: cl.rolling(m).mean() for m in v28.GRID["ma"]}
    ind["mom"] = {m: cl / cl.shift(m) - 1 for m in v28.GRID["mom"]}
    ind["bench"] = cl.pct_change(fill_method=None).mean(axis=1).add(1).cumprod()
    return ind


def run(ind, buy, score, sell, i0, i1, chand=None, cap=100.0):
    """범용 시뮬레이터. buy/sell 은 [T,N] bool, score 는 [T,N] (클수록 먼저 산다).
    chand 를 주면 샹들리에 추적손절도 청산 조건에 더한다. 회계는 v28.sim 과 같다."""
    O, H, C = (ind[k].values for k in ("op", "hi", "cl"))
    A = ind["atr"].values
    slot = cap / SLOTS
    cash, pos, eq, tr = cap, {}, [], []
    for i in range(i0, min(i1, len(C) - 1)):
        for j in list(pos):
            p = pos[j]
            if np.isnan(H[i, j]):
                continue
            p["pk"] = max(p["pk"], H[i, j])
            out = bool(sell is not None and sell[i, j])
            if chand is not None and not np.isnan(A[i, j]):
                out |= C[i, j] <= p["pk"] - chand * A[i, j]
            if out:
                px = O[i + 1, j]
                if not (px > 0):
                    continue
                cash += p["q"] * px * (1 - COST / 100)
                tr.append((px / p["e"] - 1) * 100 - COST)
                del pos[j]
        if len(pos) < SLOTS:
            nx = O[i + 1]
            ok = buy[i] & (nx > 0) & ~np.isnan(score[i])
            for j in np.argsort(-np.where(ok, score[i], -np.inf), kind="stable"):
                if len(pos) >= SLOTS or cash < slot * 0.99 or not ok[j]:
                    break
                if j in pos:
                    continue
                cash -= slot
                pos[j] = {"q": slot / nx[j], "e": nx[j], "pk": nx[j]}
        eq.append(cash + sum(p["q"] * C[i, j] for j, p in pos.items()
                             if not np.isnan(C[i, j])))
    e = pd.Series(eq)
    return e.pct_change().dropna().values, tr


def kis_signals(ind, lo):
    """원본 10종의 (buy, score, sell). 전부 '그날 종가까지'로 계산된다."""
    cl, hi = ind["cl"], ind["hi"]
    lo = lo.reindex(cl.index)
    ch = cl.diff()
    S = {}

    ma5, ma20 = cl.rolling(5).mean(), cl.rolling(20).mean()
    up = (ma5.shift(1) < ma20.shift(1)) & (ma5 > ma20)
    dn = (ma5.shift(1) > ma20.shift(1)) & (ma5 < ma20)
    S["①골든크로스 5/20"] = (up, up * 0.7, dn, None)

    r60 = cl.pct_change(60, fill_method=None)
    S["②모멘텀 60일 ±"] = (r60 >= 0.30, (0.5 + r60).clip(upper=1.0), r60 <= -0.20, None)

    # 52주 고가: 원본은 현재가 API 의 w52 고가. 종가가 '직전 250일 고가'를 넘는 것으로 옮긴다.
    w52 = hi.rolling(250).max().shift(1)
    S["③52주 신고가"] = (cl > w52, cl * 0 + 0.85, None, 3.0)

    def streak(cond):
        c = cond.values
        out = np.zeros(c.shape)
        for i in range(1, len(c)):
            out[i] = np.where(c[i], out[i - 1] + 1, 0)
        return pd.DataFrame(out, index=cond.index, columns=cond.columns)
    upd, dnd = streak(ch > 0), streak(ch < 0)
    S["④연속상승 5일"] = (upd >= 5, (0.5 + upd * 0.08).clip(upper=0.9), dnd >= 5, None)

    disp = cl / ma20 * 100
    S["⑤이격도 90/110"] = (disp < 90, ((90 - disp) / 20 + 0.5).clip(upper=1.0),
                         disp > 110, None)

    rng = (hi - lo).replace(0, np.nan)
    ratio = (cl - lo) / rng
    S["⑦강한 종가"] = (ratio >= 0.8, (0.5 + ratio * 0.4).clip(upper=1.0), None, 3.0)

    vol = cl.pct_change(fill_method=None).rolling(10).std()
    vmin = vol.rolling(10).min()
    chg = cl.pct_change(fill_method=None) * 100
    S["⑧변동성 확장"] = ((vol <= vmin * 1.1) & (chg >= 3.0), cl * 0 + 0.75, None, 3.0)

    dev = (cl / ma5 - 1) * 100
    S["⑨평균회귀 5일 ±3%"] = (dev <= -3.0, (0.5 + dev.abs() / 10).clip(upper=1.0),
                           dev >= 3.0, None)

    ma60 = cl.rolling(60).mean()
    S["⑩추세필터 MA60"] = ((cl > ma60) & (ch > 0), cl * 0 + 0.65,
                         (cl < ma60) & (ch < 0), None)
    return S


def breakout_fail(ind):
    """⑥ 돌파 실패(매도 전용): 최근 3일 고가가 그 전 25일 고가를 넘었는데 종가가 그 고점 −3% 이하."""
    hi, cl = ind["hi"], ind["cl"]
    rh = hi.rolling(3).max()
    ph = hi.shift(3).rolling(25).max()
    return (rh > ph) & (cl / rh - 1 <= -0.03)


def stats(r, tr):
    eq = (1 + pd.Series(r)).cumprod()
    return {"tot": (eq.iloc[-1] - 1) * 100, "sr": v28.sharpe(r),
            "mdd": float((eq / eq.cummax() - 1).min()) * 100, "n": len(tr),
            "avg": float(np.mean(tr)) if tr else 0.0,
            "win": 100 * float(np.mean(np.array(tr) > 0)) if tr else 0.0}


def ours(ind, cfg):
    buy = (ind["cl"] > ind["dc"][cfg["dc"]]) & (ind["cl"] > ind["ma"][cfg["ma"]])
    return buy, ind["mom"][cfg["mom"]]


def main(market="us"):
    global COST
    if market == "kr":
        COST = v28.COST = 0.23      # 국장: 수수료 0.015%×2 + 거래세 0.20%
        ind = build_kr()
        lo = load_low(ind, KR_PX)
        label = f"국장 {len(ind['codes'])}종목"
    else:
        ind = build()
        lo = load_low(ind)
        label = f"미장 {len(ind['codes'])}종목"
    idx = ind["cl"].index

    # v28 워크포워드 검증 창의 범위를 그대로 재현
    s0, s1 = idx.searchsorted(pd.Timestamp(v28.WF_START)), idx.searchsorted(pd.Timestamp(v28.WF_END))
    a0, a = s0 + v28.TRAIN, s0
    while a + v28.TRAIN + v28.TEST <= s1:
        a += v28.TEST
    a1 = a + v28.TRAIN
    print(f"[v30] 한투 전략 프리셋 10종 — {label}, 슬롯 {SLOTS}, 비용 {COST}%")
    print(f"  비교 구간 {idx[a0].date()} ~ {idx[a1 - 1].date()} (v28 워크포워드 검증 창과 동일)\n")

    # ── 정합성 검사: 범용 시뮬레이터 == v28.sim ──
    cfg = {"dc": 40, "ma": 120, "chand": 3.0, "mom": 20}
    b, sc = ours(ind, cfg)
    ok_all = b.values & ~np.isnan(ind["atr"].values)
    r_mine, t_mine = run(ind, ok_all, sc.values, None, a0, a1, chand=cfg["chand"])
    r_ref, t_ref = v28.sim(ind, cfg, SLOTS, a0, a1)
    same = len(t_mine) == len(t_ref) and np.allclose(r_mine, r_ref)
    print(f"  정합성: 범용 시뮬 vs v28.sim → 거래 {len(t_mine)} vs {len(t_ref)}, "
          f"수익률 일치 {same}")
    if not same:
        raise SystemExit("  ✗ 시뮬레이터 불일치 — 결과를 믿을 수 없다")

    rows = []
    combos = [dict(zip(v28.GRID, v)) for v in
              __import__("itertools").product(*v28.GRID.values())]
    wf = v28.walkforward(ind, SLOTS, combos, idx)
    # 거래당 평균은 비운다: 창마다 장부를 새로 시작해 **창 끝에 들고 있던 종목(대개
    # 수익 중)이 거래 목록에서 빠지고** 손절분만 남아 음수로 왜곡된다. 누적·SR·MDD 는 정확.
    rows.append(("★우리 봇(분기 재선택, 워크포워드)",
                 {k: wf[k] for k in ("tot", "sr", "mdd", "n")}
                 | {"avg": float("nan"), "win": float("nan")}))

    bench = ind["bench"].iloc[a0:a1]
    rb = bench.pct_change().dropna().values
    rows.append(("시장(유니버스 동일가중)", stats(rb, [])))

    S = kis_signals(ind, lo)
    for name, (buy, score, sell, ch) in S.items():
        bv = buy.fillna(False).values
        sv = sell.fillna(False).values if sell is not None else None
        r, t = run(ind, bv, score.values, sv, a0, a1, chand=ch)
        tag = " (+샹들리에 청산)" if ch else ""
        rows.append((name + tag, stats(r, t)))
        if sell is not None:     # 원본 청산에 추적손절을 더하면?
            r, t = run(ind, bv, score.values, sv, a0, a1, chand=3.0)
            rows.append(("   └ +샹들리에 3×ATR", stats(r, t)))

    # ⑥ 은 우리 봇 청산에 덧붙인다 — 같은 고정 설정 두 개로 비교
    bf = breakout_fail(ind).fillna(False).values
    for cfg in ({"dc": 40, "ma": 120, "chand": 3.0, "mom": 20},
                {"dc": 40, "ma": 50, "chand": 2.0, "mom": 60}):
        b, sc = ours(ind, cfg)
        bv = b.values & ~np.isnan(ind["atr"].values)
        tag = f"dc{cfg['dc']}/ma{cfg['ma']}/ch{cfg['chand']}/mom{cfg['mom']}"
        r, t = run(ind, bv, sc.values, None, a0, a1, chand=cfg["chand"])
        rows.append((f"우리 고정 {tag}", stats(r, t)))
        r, t = run(ind, bv, sc.values, bf, a0, a1, chand=cfg["chand"])
        rows.append(("   └ +⑥돌파실패 청산", stats(r, t)))

    print(f"\n  {'전략':<34}{'누적%':>8}{'Sharpe':>8}{'MDD%':>8}{'거래':>6}{'거래당%':>8}{'승률%':>7}")
    print("  " + "-" * 79)
    for name, s in rows:
        win = "" if np.isnan(s["win"]) else f"{s['win']:.0f}"
        avg = "" if np.isnan(s["avg"]) else f"{s['avg']:.2f}"
        print(f"  {name:<34}{s['tot']:>8.1f}{s['sr']:>8.2f}{s['mdd']:>8.1f}"
              f"{s['n']:>6}{avg:>8}{win:>7}")


if __name__ == "__main__":
    import sys
    main("kr" if "--kr" in sys.argv else "us")
