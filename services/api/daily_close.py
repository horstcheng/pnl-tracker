import json
import logging
import os
import urllib.request
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import gspread
import requests
import yfinance as yf
from google.oauth2.service_account import Credentials

from packages.pnl.engine import (
    PositionResult,
    Transaction,
    TxType,
    compute_position_weighted_avg_cost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_google_sheet_client() -> gspread.Client:
    """Create a gspread client using service account JSON from env."""
    sa_json = os.environ["GOOGLE_SA_JSON"]
    sa_info = json.loads(sa_json)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
    ]
    credentials = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(credentials)


def read_trades_from_sheet(client: gspread.Client, sheet_id: str) -> List[dict]:
    """
    Read trades from the 'trades' tab of the Google Sheet.

    Uses get_all_values() to preserve formatted strings (e.g., '0050' stays '0050').
    """
    spreadsheet = client.open_by_key(sheet_id)
    worksheet = spreadsheet.worksheet("trades")

    # Get all values as strings (preserves leading zeros)
    all_values = worksheet.get_all_values()

    if not all_values:
        return []

    # First row is headers
    headers = [h.strip() for h in all_values[0]]

    # Build list of dicts from remaining rows
    records = []
    for row_idx, row in enumerate(all_values[1:], start=2):
        record = {}
        for col_idx, header in enumerate(headers):
            value = row[col_idx] if col_idx < len(row) else ""
            record[header] = value
        records.append(record)

    return records


def validate_symbol(symbol: str, row_context: str) -> str:
    """Validate and return symbol, raising error if invalid."""
    if not isinstance(symbol, str):
        raise ValueError(f"Symbol must be a string, got {type(symbol).__name__}: {symbol} ({row_context})")
    symbol = symbol.strip()
    if not symbol:
        raise ValueError(f"Symbol cannot be empty ({row_context})")
    return symbol


def get_usd_twd_rate() -> Decimal:
    """Fetch USD/TWD exchange rate from a public API."""
    try:
        resp = requests.get(
            "https://api.exchangerate-api.com/v4/latest/USD",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = data["rates"]["TWD"]
        logger.info(f"USD/TWD rate: {rate}")
        return Decimal(str(rate))
    except requests.RequestException as e:
        logger.error(f"Failed to fetch USD/TWD rate: {e}")
        raise


def normalize_ticker(symbol: str, asset_ccy: str) -> str:
    """
    Normalize a symbol to its primary yfinance ticker.

    Args:
        symbol: The trading symbol as a string (must preserve leading zeros and letters).
        asset_ccy: The asset currency (e.g., "TWD", "USD").

    Returns:
        The normalized yfinance ticker string.

    Examples:
        normalize_ticker("00983A", "TWD") -> "00983A.TW"
        normalize_ticker("2330", "TWD") -> "2330.TW"
        normalize_ticker("AAPL", "USD") -> "AAPL"
    """
    # Ensure symbol is treated as string, preserving leading zeros and letters
    symbol = str(symbol).strip()

    # For TWD symbols without an existing suffix, append .TW
    if asset_ccy == "TWD" and "." not in symbol:
        return f"{symbol}.TW"
    return symbol


def get_yfinance_tickers_to_try(symbol: str, asset_ccy: str) -> List[str]:
    """
    Return list of yfinance ticker symbols to try for a given symbol.

    Uses normalize_ticker() for the primary ticker, then adds .TWO fallback for TWD.
    """
    symbol = str(symbol).strip()
    primary = normalize_ticker(symbol, asset_ccy)

    # For TWD symbols without existing suffix, also try OTC market (.TWO) as fallback
    if asset_ccy == "TWD" and "." not in symbol:
        return [primary, f"{symbol}.TWO"]
    return [primary]


def try_fetch_price(yf_symbol: str) -> Tuple[bool, Optional[Decimal]]:
    """
    Try to fetch close price for a single yfinance symbol.

    Returns (success, price) tuple.
    """
    try:
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return False, None
        close_series = hist["Close"].dropna()
        if close_series.empty:
            return False, None
        close_price = close_series.iloc[-1]
        return True, Decimal(str(close_price))
    except Exception as e:
        logger.debug(f"Failed to fetch {yf_symbol}: {e}")
        return False, None


def try_fetch_price_with_prev(
    yf_symbol: str,
    original_symbol: Optional[str] = None,
) -> Tuple[bool, Optional[Decimal], Optional[Decimal]]:
    """
    Try to fetch current and previous trading day close prices.

    Fetches 7 days of history to ensure we get at least two valid close prices
    even during holidays or market closures.

    Args:
        yf_symbol: The yfinance ticker symbol to fetch.
        original_symbol: The original symbol (for debug logging).

    Returns (success, today_close, prev_close) tuple.
    prev_close may be None even if success is True (insufficient history).
    """
    display_symbol = original_symbol or yf_symbol
    try:
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="7d")
        if hist.empty:
            return False, None, None
        close_series = hist["Close"].dropna()
        if close_series.empty:
            return False, None, None
        today_close = Decimal(str(close_series.iloc[-1]))
        prev_close = None
        if len(close_series) >= 2:
            prev_close = Decimal(str(close_series.iloc[-2]))
        else:
            # Debug logging for missing prev_close
            last_3_dates = list(close_series.index[-3:]) if len(close_series) >= 1 else []
            last_3_closes = [float(close_series.iloc[i]) for i in range(-min(3, len(close_series)), 0)] if len(close_series) >= 1 else []
            logger.warning(
                f"Missing prev_close debug: symbol={display_symbol}, ticker={yf_symbol}, "
                f"history_len={len(close_series)}, "
                f"last_dates={[str(d.date()) for d in last_3_dates]}, "
                f"last_closes={last_3_closes}"
            )
        return True, today_close, prev_close
    except Exception as e:
        logger.debug(f"Failed to fetch {yf_symbol}: {e}")
        return False, None, None


def get_close_prices(
    symbols_with_ccy: List[Tuple[str, str]]
) -> Tuple[Dict[str, Decimal], Dict[str, Decimal], Dict[str, List[Tuple[str, str]]], List[str]]:
    """
    Fetch close prices for all symbols using yfinance.

    Returns:
        - prices: dict of symbol -> Decimal price (today's close)
        - prev_prices: dict of symbol -> Decimal price (previous trading day close)
        - lookup_attempts: dict of symbol -> list of (ticker, result) for missing symbols
        - missing_prev_close: list of symbols missing previous close data
    """
    prices: Dict[str, Decimal] = {}
    prev_prices: Dict[str, Decimal] = {}
    lookup_attempts: Dict[str, List[Tuple[str, str]]] = {}
    missing_prev_close: List[str] = []

    for symbol, asset_ccy in symbols_with_ccy:
        tickers_to_try = get_yfinance_tickers_to_try(symbol, asset_ccy)
        attempts: List[Tuple[str, str]] = []
        found = False

        for yf_symbol in tickers_to_try:
            success, today_price, prev_price = try_fetch_price_with_prev(yf_symbol, original_symbol=symbol)
            if success and today_price is not None:
                prices[symbol] = today_price
                if prev_price is not None:
                    prev_prices[symbol] = prev_price
                else:
                    missing_prev_close.append(symbol)
                    logger.warning(f"No previous close for {symbol} ({yf_symbol})")
                logger.info(f"Price for {symbol} ({yf_symbol}): today={today_price}, prev={prev_price}")
                attempts.append((yf_symbol, "ok"))
                found = True
                break
            else:
                attempts.append((yf_symbol, "fail"))
                logger.debug(f"No price data for {yf_symbol}")

        if not found:
            logger.warning(f"No price data for {symbol} after trying: {tickers_to_try}")
            lookup_attempts[symbol] = attempts

    return prices, prev_prices, lookup_attempts, missing_prev_close


def build_transactions(trades: List[dict]) -> Dict[Tuple[str, str, str], List[Transaction]]:
    """Group trades by (user_id, symbol, asset_ccy) and build Transaction objects."""
    grouped: Dict[Tuple[str, str, str], List[Transaction]] = defaultdict(list)

    for idx, trade in enumerate(trades, start=2):
        row_context = f"row {idx}"

        user_id = str(trade["user_id"]).strip()
        symbol = validate_symbol(trade["symbol"], row_context)
        asset_ccy = str(trade["asset_ccy"]).strip().upper()
        side = str(trade["side"]).strip().upper()
        quantity = Decimal(str(trade["quantity"]))
        price = Decimal(str(trade["price"]))
        fee = Decimal(str(trade["fee"]))
        trade_date = str(trade["trade_date"]).strip()

        tx_type = TxType.BUY if side == "BUY" else TxType.SELL

        tx = Transaction(
            trade_date=trade_date,
            tx_type=tx_type,
            quantity=quantity,
            price=price,
            fee=fee,
        )

        key = (user_id, symbol, asset_ccy)
        grouped[key].append(tx)

    return grouped


def compute_all_pnl(
    grouped_txs: Dict[Tuple[str, str, str], List[Transaction]],
    prices: Dict[str, Decimal],
    usd_twd: Decimal,
) -> Tuple[Dict[str, Decimal], Decimal, List[Tuple[str, Decimal]], Dict[str, Tuple[Decimal, str]]]:
    """
    Compute P&L for all positions.

    Returns:
        - user_pnl: dict of user_id -> total P&L in TWD
        - total_pnl: overall total P&L in TWD
        - symbol_pnl: list of (symbol, pnl_twd) sorted by absolute value
        - symbol_positions: dict of symbol -> (quantity, asset_ccy) for Day P&L calculation
    """
    user_pnl: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    symbol_pnl: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    symbol_positions: Dict[str, Tuple[Decimal, str]] = {}
    total_pnl = Decimal("0")

    for (user_id, symbol, asset_ccy), transactions in grouped_txs.items():
        if symbol not in prices:
            logger.warning(f"Skipping {symbol} - no price available")
            continue

        close_price = prices[symbol]

        try:
            result: PositionResult = compute_position_weighted_avg_cost(
                transactions=transactions,
                close_price=close_price,
            )
        except ValueError as e:
            logger.error(f"Error computing P&L for {user_id}/{symbol}: {e}")
            continue

        pnl_in_ccy = result.realized_pnl + result.unrealized_pnl

        if asset_ccy == "TWD":
            pnl_twd = pnl_in_ccy
        elif asset_ccy == "USD":
            pnl_twd = pnl_in_ccy * usd_twd
        else:
            logger.warning(f"Unknown currency {asset_ccy}, treating as TWD")
            pnl_twd = pnl_in_ccy

        user_pnl[user_id] += pnl_twd
        symbol_pnl[symbol] += pnl_twd
        total_pnl += pnl_twd

        # Aggregate position quantity per symbol for Day P&L
        if symbol in symbol_positions:
            existing_qty, existing_ccy = symbol_positions[symbol]
            symbol_positions[symbol] = (existing_qty + result.quantity, existing_ccy)
        else:
            symbol_positions[symbol] = (result.quantity, asset_ccy)

        logger.info(
            f"{user_id}/{symbol}/{asset_ccy}: "
            f"realized={result.realized_pnl}, unrealized={result.unrealized_pnl}, "
            f"pnl_twd={pnl_twd:.2f}, qty={result.quantity}"
        )

    sorted_symbols = sorted(
        symbol_pnl.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return dict(user_pnl), total_pnl, sorted_symbols, symbol_positions


def compute_day_pnl(
    symbol_positions: Dict[str, Tuple[Decimal, str]],
    prices: Dict[str, Decimal],
    prev_prices: Dict[str, Decimal],
    usd_twd: Decimal,
) -> Tuple[Decimal, List[Tuple[str, Decimal]]]:
    """
    Compute Day P&L for all symbols.

    Day P&L = (today_close - prev_close) * current_position_quantity

    Args:
        symbol_positions: dict of symbol -> (quantity, asset_ccy)
        prices: dict of symbol -> today's close price
        prev_prices: dict of symbol -> previous trading day close price
        usd_twd: USD/TWD exchange rate

    Returns:
        - total_day_pnl: total Day P&L in TWD
        - symbol_day_pnl: list of (symbol, day_pnl_twd) sorted by absolute value
    """
    symbol_day_pnl: Dict[str, Decimal] = {}
    total_day_pnl = Decimal("0")

    for symbol, (quantity, asset_ccy) in symbol_positions.items():
        if symbol not in prices or symbol not in prev_prices:
            # Skip symbols without both prices (Day P&L = 0 for these)
            logger.info(f"Day P&L for {symbol}: skipped (missing price data)")
            continue

        if quantity == Decimal("0"):
            # No position, no Day P&L
            continue

        today_close = prices[symbol]
        prev_close = prev_prices[symbol]
        day_pnl_native = (today_close - prev_close) * quantity

        if asset_ccy == "TWD":
            day_pnl_twd = day_pnl_native
        elif asset_ccy == "USD":
            day_pnl_twd = day_pnl_native * usd_twd
        else:
            day_pnl_twd = day_pnl_native

        symbol_day_pnl[symbol] = day_pnl_twd
        total_day_pnl += day_pnl_twd

        logger.info(
            f"Day P&L for {symbol}: ({today_close} - {prev_close}) * {quantity} = "
            f"{day_pnl_native} {asset_ccy} = {day_pnl_twd:.2f} TWD"
        )

    sorted_day_pnl = sorted(
        symbol_day_pnl.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return total_day_pnl, sorted_day_pnl


def format_slack_message(
    today: str,
    usd_twd: Decimal,
    total_pnl: Decimal,
    user_pnl: Dict[str, Decimal],
    top_symbols: List[Tuple[str, Decimal]],
    symbols_count: int,
    missing_symbols: List[str],
    lookup_attempts: Dict[str, List[Tuple[str, str]]],
    total_day_pnl: Decimal,
    top_day_symbols: List[Tuple[str, Decimal]],
    missing_prev_close: List[str],
) -> str:
    """Format the Slack message."""
    top5 = top_symbols[:5]
    top5_subtotal = sum(pnl for _, pnl in top5)
    others_subtotal = total_pnl - top5_subtotal

    lines = [
        f"*Daily P&L Report - {today}*",
        "",
        f"USD/TWD: {usd_twd:.2f}",
        f"Symbols counted: {symbols_count}",
        "",
        "*User P&L Ranking (Top 3):*",
    ]

    sorted_users = sorted(user_pnl.items(), key=lambda x: x[1], reverse=True)
    for i, (user_id, pnl) in enumerate(sorted_users[:3], 1):
        lines.append(f"  {i}. {user_id}: {pnl:+,.0f} TWD")

    lines.append("")
    lines.append("*Top 5 Symbols (Total):*")

    for i, (symbol, pnl) in enumerate(top5, 1):
        lines.append(f"  {i}. {symbol}: {pnl:+,.0f} TWD")

    lines.append("")
    lines.append(f"Top5 subtotal: {top5_subtotal:+,.0f} TWD")
    lines.append(f"Others subtotal: {others_subtotal:+,.0f} TWD")
    lines.append(f"Grand total: {total_pnl:+,.0f} TWD")

    # Day P&L section (Today)
    lines.append("")
    lines.append(f"*Total P&L (Today): {total_day_pnl:+,.0f} TWD*")

    lines.append("")
    lines.append("*Top 5 Symbols (Today):*")
    top5_day = top_day_symbols[:5]
    for i, (symbol, pnl) in enumerate(top5_day, 1):
        lines.append(f"  {i}. {symbol}: {pnl:+,.0f} TWD")

    # Day P&L warnings
    if missing_prev_close:
        lines.append("")
        lines.append(f"Day P&L warnings: missing previous close for {', '.join(sorted(missing_prev_close))}")
        lines.append("Day P&L note: missing prev_close symbols are treated as 0 (excluded from today's move).")

    lines.append("")
    missing_str = ", ".join(missing_symbols) if missing_symbols else "None"
    lines.append(f"Missing prices (Total): {missing_str}")

    if lookup_attempts:
        attempts_parts = []
        for sym in sorted(lookup_attempts.keys()):
            attempts = lookup_attempts[sym]
            attempt_str = ", ".join(f"{t}={r}" for t, r in attempts)
            attempts_parts.append(f"{sym}: {attempt_str}")
        lines.append(f"Price lookup attempts (missing only): {'; '.join(attempts_parts)}")

    return "\n".join(lines)


def post_to_slack(text: str) -> None:
    """Post message to Slack via webhook."""
    url = os.environ["SLACK_WEBHOOK_URL"]
    payload = {"text": text}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


def main() -> None:
    """Main entry point."""
    logger.info("Starting daily close process")

    sheet_id = os.environ["GOOGLE_SHEET_ID"]

    client = get_google_sheet_client()
    logger.info("Connected to Google Sheets")

    trades = read_trades_from_sheet(client, sheet_id)
    logger.info(f"Read {len(trades)} trades from sheet")

    if not trades:
        logger.info("No trades found, exiting")
        return

    usd_twd = get_usd_twd_rate()

    symbols_with_ccy = list({
        (validate_symbol(t["symbol"], f"row {i+2}"), str(t["asset_ccy"]).strip().upper())
        for i, t in enumerate(trades)
    })
    prices, prev_prices, lookup_attempts, missing_prev_close = get_close_prices(symbols_with_ccy)
    logger.info(f"Fetched prices for {len(prices)} symbols")

    all_symbols = {sym for sym, _ in symbols_with_ccy}
    missing_symbols = sorted(all_symbols - set(prices.keys()))

    grouped_txs = build_transactions(trades)

    user_pnl, total_pnl, top_symbols, symbol_positions = compute_all_pnl(grouped_txs, prices, usd_twd)

    # Compute Day P&L
    total_day_pnl, top_day_symbols = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)
    logger.info(f"Day P&L calculated: {total_day_pnl:.2f} TWD")

    today = date.today().isoformat()
    message = format_slack_message(
        today, usd_twd, total_pnl, user_pnl, top_symbols,
        symbols_count=len(prices),
        missing_symbols=missing_symbols,
        lookup_attempts=lookup_attempts,
        total_day_pnl=total_day_pnl,
        top_day_symbols=top_day_symbols,
        missing_prev_close=missing_prev_close,
    )

    logger.info("Posting to Slack")
    post_to_slack(message)
    logger.info("Daily close completed successfully")


if __name__ == "__main__":
    main()
