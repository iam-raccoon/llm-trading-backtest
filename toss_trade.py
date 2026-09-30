# -*- coding: utf-8 -*-
"""토스증권 Open API 실행 어댑터 — 조회 + 주문. **기본값은 주문 금지.**

━━━ 이 파일의 첫 번째 목적은 '사고 방지'다 ━━━
토스 Open API 에는 **샌드박스/모의투자 환경이 없다**(명세 확인). 즉 코드를 시험하는
순간이 곧 실주문이다. 그래서 안전장치를 켜는 쪽이 아니라 **끄는 쪽**을 어렵게 만든다:

  1. `TOSS_LIVE=1` 환경변수가 없으면 **주문 함수가 아예 실행되지 않는다**(DryRun 반환).
  2. 그 위에 호출 인자로 `confirm=True` 를 명시해야 한다. 둘 다여야 나간다.
  3. 1회 주문 금액이 `TOSS_MAX_ORDER_KRW`(기본 100,000원)를 넘으면 거부한다.
  4. 등록 IP가 바뀌면 주문을 막는다 — 토스는 IP 화이트리스트를 쓰므로 IP가 바뀌면
     인증이 엉뚱한 에러로 실패한다. 조용히 실패하느니 멈추는 게 낫다.

━━━ 키 취급 ━━━
`.env`(chmod 600, gitignore)에서만 읽는다. **어떤 경로로도 출력하지 않는다** —
로그·예외 메시지·리포지토리 어디에도 남기지 않는다. 진단이 필요하면 길이와 앞뒤
두 글자만 쓴다(`mask()`).

━━━ 실측으로 확인된 사실 (2026-08-31) ━━━
· 인증: OAuth2 client_credentials, 토큰 유효 86399초(24h).
· 계좌 헤더는 `X-Tossinvest-Account: <accountSeq>` — **계좌번호가 아니라 정수 seq**다.
  계좌번호를 넣으면 전부 `account-not-found` 가 난다(실제로 겪었다).
· **수수료 API 실측**: `GET /api/v1/commissions` → KR `0.00015`(0.015%), US `0.001`.
  공개 자료와 일치. 왕복 비용은 여기에 매도 거래세 0.20%(2026)를 더해 **0.23%**.
  → `kr_costs.py` 참조. 가정이 아니라 API 가 준 값이다.

사용:
    from toss_trade import Toss
    t = Toss()                       # 조회 전용
    t.price("005930")
    t.holdings()
    t.buy("005930", 1)               # TOSS_LIVE 없으면 DryRun
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://openapi.tossinvest.com"
ENV_PATHS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
             os.path.expanduser("~/llm-trading-backtest/.env")]

MAX_ORDER_KRW = int(os.environ.get("TOSS_MAX_ORDER_KRW", "100000"))
EXPECT_IP = os.environ.get("TOSS_EXPECT_IP", "")   # 비우면 IP 검사 안 함


class TossError(RuntimeError):
    pass


class DryRun(dict):
    """실제로 나가지 않은 주문. 딕셔너리처럼 쓰되 정체가 드러나게."""

    def __repr__(self):
        return f"DryRun({dict.__repr__(self)})"


def mask(s):
    if not s:
        return "(없음)"
    return f"길이 {len(s)}, {s[:2]}…{s[-2:]}"


def _load_env():
    for p in ENV_PATHS:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        return True
    return False


def public_ip(timeout=10):
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=timeout) as r:
            return r.read().decode().strip()
    except Exception:
        return ""


class Toss:
    def __init__(self, account_seq=None):
        _load_env()
        self._id = os.environ.get("TOSS_CLIENT_ID", "")
        self._sec = os.environ.get("TOSS_CLIENT_SECRET", "")
        if not self._id or not self._sec:
            raise TossError("TOSS_CLIENT_ID/SECRET 가 없다 (.env 확인). "
                            "값은 출력하지 않는다.")
        self._tok = None
        self._exp = 0
        self.seq = account_seq if account_seq is not None else self._first_seq()

    # ── 인증 ────────────────────────────────────────────────
    def _token(self):
        if self._tok and time.time() < self._exp - 60:
            return self._tok
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self._id, "client_secret": self._sec}).encode()
        req = urllib.request.Request(
            BASE + "/oauth2/token", data=body, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        # ⚠️ DNS 가 간헐적으로 튕겨 **토큰 발급부터** 죽어 봇이 시작도 못 한 날이 있다
        # (2026-09-29/30). 여기에도 재시도를 넣는다.
        j = None
        for k in range(4):
            try:
                with urllib.request.urlopen(req, timeout=25) as r:
                    j = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                # ⚠️ 예외 본문에 키가 실릴 일은 없지만, 혹시 몰라 코드만 올린다.
                raise TossError(f"토큰 발급 실패 HTTP {e.code}. "
                                f"등록 IP({EXPECT_IP or '미지정'}) 불일치 가능") from None
            except Exception:
                if k == 3:
                    raise
                time.sleep(2 * (k + 1))
        self._tok = j["access_token"]
        self._exp = time.time() + int(j.get("expires_in", 3600))
        return self._tok

    def _call(self, method, path, params=None, body=None, need_acct=True):
        url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
        h = {"Accept": "application/json",
             "Authorization": f"Bearer {self._token()}"}
        if need_acct and self.seq is not None:
            h["X-Tossinvest-Account"] = str(self.seq)   # ★ accountNo 아님
        data = None
        if body is not None:
            data = json.dumps(body).encode()
        # ⚠️ 본문이 없는 POST(주문 취소 등)에도 Content-Type 이 필요하다.
        # 이걸 빼먹어 취소가 `415 unsupported-content-type` 으로 실패했고,
        # 시험 주문이 계좌에 남았다(2026-08-31). 주문을 내는 코드가 취소를
        # 못 하면 그건 안전장치가 아니라 위험이다.
        if method in ("POST", "PUT", "PATCH"):
            h["Content-Type"] = "application/json"
            if data is None:
                data = b"{}"
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        # ⚠️ 젯슨 DNS(테일스케일 100.100.100.100)가 부하 중 간헐적으로 튕긴다.
        # 이 때문에 7일 중 4일 봇이 통째로 멈췄다. 네트워크 계열 실패와 429 는
        # 재시도한다. 단 **주문(POST /orders)은 재시도하지 않는다** — 첫 요청이
        # 실제로는 접수됐는데 응답만 못 받은 경우 중복 주문이 된다.
        retry_ok = not (method == "POST" and path.startswith("/api/v1/orders"))
        tries = 4 if retry_ok else 1
        for k in range(tries):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read() or b"{}")
            except urllib.error.HTTPError as e:
                if e.code == 429 and k < tries - 1:
                    time.sleep(2 * (k + 1))
                    continue
                # ★ 401 token-revoked: 토스는 새 토큰을 발급하면 이전 토큰을 죽인다.
                #   한 실행에서 Toss 객체를 두 번 만들어 서로 죽이는 바람에 **매수
                #   주문이 거부**됐다(2026-09-29, 현금 3일간 유휴). 객체는 하나만
                #   쓰도록 고쳤지만, 남은 경우(다른 프로세스가 토큰을 새로 받음)를
                #   대비해 여기서도 복구한다. **요청이 거부된 것이므로** 주문이라도
                #   재시도해도 중복되지 않는다.
                if e.code == 401 and not getattr(self, "_retried_401", False):
                    body_txt = ""
                    try:
                        body_txt = e.read().decode()
                    except Exception:
                        pass
                    if "token-revoked" in body_txt or "revoked" in body_txt:
                        self._retried_401 = True
                        self._tok, self._exp = None, 0
                        h["Authorization"] = f"Bearer {self._token()}"
                        req2 = urllib.request.Request(url, data=data, headers=h,
                                                     method=method)
                        with urllib.request.urlopen(req2, timeout=30) as r:
                            self._retried_401 = False
                            return json.loads(r.read() or b"{}")
                    e2 = urllib.error.HTTPError(e.url, e.code, body_txt, e.hdrs, None)
                    return self._raise_http(e2, method, path)
                return self._raise_http(e, method, path)
            except (urllib.error.URLError, TimeoutError, OSError):
                if k < tries - 1:
                    time.sleep(2 * (k + 1))
                    continue
                raise

    def _raise_http(self, e, method, path):
        try:
            j = json.loads(e.read() or b"{}")
        except Exception:
            j = {}
        err = (j.get("error") or {}) if isinstance(j, dict) else {}
        raise TossError(f"HTTP {e.code} {err.get('code','')} "
                        f"{err.get('message','')} ({method} {path})") from None

    def _first_seq(self):
        r = self._call("GET", "/api/v1/accounts", need_acct=False).get("result") or []
        if not r:
            raise TossError("계좌가 없다")
        return r[0]["accountSeq"]

    # ── 조회 ────────────────────────────────────────────────
    def accounts(self):
        return self._call("GET", "/api/v1/accounts", need_acct=False).get("result")

    def commission(self, market="KR"):
        """매매 수수료율(편도, 비율). 가정하지 말고 이걸 쓴다."""
        for x in self._call("GET", "/api/v1/commissions",
                            {"market": market}).get("result") or []:
            if x.get("marketCountry") == market:
                return float(x["commissionRate"])
        return None

    def price(self, symbols, market="KR"):
        if isinstance(symbols, (list, tuple)):
            symbols = ",".join(symbols)
        return self._call("GET", "/api/v1/prices",
                          {"symbols": symbols, "market": market}).get("result")

    def buying_power(self, currency="KRW"):
        """주문 가능 현금. ⚠️ 파라미터가 `currency` 다 — `market` 을 넣으면 400 이
        나는데 응답 본문이 비어 있어 원인이 안 보인다(실제로 겪었다)."""
        return self._call("GET", "/api/v1/buying-power",
                          {"currency": currency}).get("result")

    def holdings(self, market="KR"):
        return self._call("GET", "/api/v1/holdings", {"market": market}).get("result")

    def orders(self, status="OPEN", **params):
        """주문 목록. ⚠️ `status` 는 **필수**다(OPEN/CLOSED). 빠뜨리면 400 이
        나는데 응답 본문이 비어 원인이 안 보인다. 기본값을 둬서 잊을 수 없게 한다.
        OPEN 은 전량 반환이라 페이지네이션이 필요 없다."""
        params["status"] = status
        return self._call("GET", "/api/v1/orders", params).get("result")

    def cancel_all(self, market="KR"):
        """미체결 전량 취소 — **킬스위치**. 이상 시 이것부터 부른다."""
        out = self.orders(status="OPEN", market=market)
        lst = out if isinstance(out, list) else (out or {}).get("orders") or []
        done = []
        for x in lst:
            oid = x.get("orderId")
            if not oid:
                continue
            try:
                self.cancel(oid, confirm=True)
                done.append(oid)
            except Exception:
                pass
        return {"found": len(lst), "cancelled": len(done)}

    # ── 주문 (기본 금지) ─────────────────────────────────────
    def _guard(self, symbol, qty, price, side):
        """주문 전 관문. 하나라도 걸리면 실주문이 안 나간다.

        ⚠️ qty 를 문자열("0.1")로 넘기면 `price * qty` 가 문자열 반복이 되어
        비교에서 TypeError 가 났다(2026-08-31 실측). 숫자로 강제한다."""
        reasons = []
        if os.environ.get("TOSS_LIVE") != "1":
            reasons.append("TOSS_LIVE!=1")
        try:
            qty = float(qty)
        except (TypeError, ValueError):
            reasons.append(f"수량이 숫자가 아니다: {qty!r}")
            qty = 0.0
        est = (float(price) if price else 0.0) * qty
        if price and est > MAX_ORDER_KRW:
            reasons.append(f"1회 한도 초과 {est:,.0f} > {MAX_ORDER_KRW:,}")
        if EXPECT_IP:
            now = public_ip()
            if now and now != EXPECT_IP:
                reasons.append(f"공인 IP 변경 {EXPECT_IP} → {now}")
        return reasons

    def order(self, symbol, qty=None, side="BUY", price=None, confirm=False,
              market="KR", amount=None):
        """수량 기반(`qty`) 또는 **금액 기반(`amount`)** 주문.

        ⚠️ **`amount` 는 US MARKET 전용**이다(명세: "금액 기반 주문 (US MARKET 전용)").
        소수점 매수는 이 경로로만 된다 — 국내는 1주 단위이고 소수점 수량은 400 이다
        (2026-08-31 실측: 1주 접수 / 0.1주 400).

        ⚠️ 금액을 수량 자리에 넘기면 참사가 난다. $24 를 quantity 로 보내면
        24주(=$4,400)를 사려 든다. 실제로 그렇게 짜여 있었고 실주문 직전에 잡았다.
        그래서 **둘 중 정확히 하나만** 받도록 강제한다.

        `confirm=True` 와 `TOSS_LIVE=1` 이 **둘 다** 있어야 실제로 나간다."""
        if (qty is None) == (amount is None):
            raise TossError("qty 와 amount 중 정확히 하나만 지정할 것")
        if amount is not None and market != "US":
            raise TossError("금액 기반 주문은 US 전용이다 (국내는 1주 단위)")
        if amount is not None and price is not None:
            raise TossError("금액 기반 주문은 시장가만 된다 (price 불가)")

        # 한도 검사: 금액 주문은 금액 자체가 노출액이다
        est = float(amount) if amount is not None else None
        reasons = (self._guard(symbol, qty, price, side) if qty is not None
                   else self._guard_amount(est))
        if not confirm:
            reasons.append("confirm=False")
        if reasons:
            return DryRun({"would": {"symbol": symbol, "qty": qty,
                                     "amount": amount, "side": side,
                                     "price": price, "market": market},
                           "blocked_by": reasons})

        body = {"symbol": symbol, "market": market, "side": side}
        if amount is not None:
            body["orderType"] = "MARKET"
            body["orderAmount"] = f"{float(amount):.2f}"
        else:
            q = float(qty)
            if q == int(q):
                body["quantity"] = str(int(q))
            else:
                # 소수점 6자리까지만 허용. **내림** — 반올림하면 보유량을 넘겨 거부된다.
                import math as _m
                qf = _m.floor(q * 1_000_000) / 1_000_000
                body["quantity"] = f"{qf:.6f}".rstrip("0").rstrip(".")
            body["orderType"] = "LIMIT" if price else "MARKET"
            if price:
                body["price"] = str(price)
        return self._call("POST", "/api/v1/orders", body=body)

    def _guard_amount(self, usd):
        """금액 기반 주문용 관문. 한도는 원화 기준이므로 환산해 비교한다."""
        reasons = []
        if os.environ.get("TOSS_LIVE") != "1":
            reasons.append("TOSS_LIVE!=1")
        if usd is None or usd <= 0:
            reasons.append(f"금액이 잘못됐다: {usd!r}")
        elif usd * 1400 > MAX_ORDER_KRW:      # 대략 환산(보수적으로 높게)
            reasons.append(f"1회 한도 초과 약 {usd*1400:,.0f}원 > {MAX_ORDER_KRW:,}")
        if EXPECT_IP:
            now = public_ip()
            if now and now != EXPECT_IP:
                reasons.append(f"공인 IP 변경 {EXPECT_IP} → {now}")
        return reasons

    def buy(self, symbol, qty=None, price=None, confirm=False, market="KR",
            amount=None):
        return self.order(symbol, qty, "BUY", price, confirm, market, amount)

    def sell(self, symbol, qty=None, price=None, confirm=False, market="KR",
             amount=None):
        return self.order(symbol, qty, "SELL", price, confirm, market, amount)

    def order_detail(self, order_id):
        return self._call("GET", f"/api/v1/orders/{order_id}").get("result")

    def wait_fill(self, order_id, timeout=90, poll=3):
        """체결될 때까지 기다렸다가 **실제 체결가·수량**을 돌려준다.

        ⚠️ 이게 없어서 상태파일에 **신호 기준일 종가**를 진입가로 적었다.
        2026-08-31 실측: TEAM 기록 $185.62 vs 실제 체결 $194.97(+5.04%),
        MSTR 기록 $137.40 vs 실제 $127.21(−7.42%). 주말 갭 때문이다.
        진입가가 틀리면 **트레일링 손절선도 틀린다** — 기록만의 문제가 아니다.

        반환 {"qty", "fill_price", "amount", "status"} 또는 None."""
        import time as _t
        end = _t.time() + timeout
        last = None
        while _t.time() < end:
            try:
                d = self.order_detail(order_id) or {}
            except Exception:
                d = {}
            last = d
            st = str(d.get("status") or "").upper()
            q = d.get("quantity")
            if st == "FILLED" and q:
                qf = float(q)
                amt = float(d.get("orderAmount") or 0) or None
                px = (amt / qf) if (amt and qf) else (
                    float(d["price"]) if d.get("price") else None)
                return {"qty": qf, "fill_price": px, "amount": amt, "status": st}
            if st in ("CANCELED", "CANCELLED", "REJECTED"):
                return {"qty": 0.0, "fill_price": None, "amount": None, "status": st}
            _t.sleep(poll)
        return {"qty": None, "fill_price": None, "amount": None,
                "status": str((last or {}).get("status") or "TIMEOUT")}

    def cancel(self, order_id, confirm=False):
        if os.environ.get("TOSS_LIVE") != "1" or not confirm:
            return DryRun({"would_cancel": order_id})
        return self._call("POST", f"/api/v1/orders/{order_id}/cancel")


def main():
    t = Toss()
    print(f"  계좌 seq={t.seq}")
    c = t.commission("KR")
    print(f"  수수료(편도) {c:.5f} = {c*100:.3f}%  → 왕복+거래세 "
          f"{c*2*100 + 0.20:.3f}%")
    print(f"  삼성전자 {t.price('005930')}")
    h = t.holdings()
    print(f"  보유주식 평가액 {h.get('marketValue', {}).get('amount', {}).get('krw')}원")
    bp = t.buying_power("KRW")
    print(f"  주문가능 현금  {bp.get('cashBuyingPower')}원")
    print(f"\n  주문 안전장치 시험 (실제로 안 나간다):")
    print(f"   {t.buy('005930', 1)}")
    print(f"\n  TOSS_LIVE={os.environ.get('TOSS_LIVE', '(없음)')}  "
          f"1회 한도 {MAX_ORDER_KRW:,}원  등록IP {EXPECT_IP or '(검사 안 함)'}")


if __name__ == "__main__":
    main()
