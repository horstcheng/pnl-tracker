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


def get_yfinance_tickers_to_try(symbol: str, asset_ccy: str) -> List[str]:
    """
    Return list of yfinance ticker symbols to try for a given symbol.

    For TWD symbols without a dot, try .TW then .TWO (OTC market).
    """
    if asset_ccy == "TWD" and "." not in symbol:
        return [f"{symbol}.TW", f"{symbol}.TWO"]
    return [symbol]


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


def get_close_prices(
    symbols_with_ccy: List[Tuple[str, str]]
) -> Tuple[Dict[str, Decimal], Dict[str, List[Tuple[str, str]]]]:
    """
    Fetch close prices for all symbols using yfinance.

    Returns:
        - prices: dict of symbol -> Decimal price
        - lookup_attempts: dict of symbol -> list of (ticker, result) for missing symbols
    """
    prices: Dict[str, Decimal] = {}
    lookup_attempts: Dict[str, List[Tuple[str, str]]] = {}

    for symbol, asset_ccy in symbols_with_ccy:
        tickers_to_try = get_yfinance_tickers_to_try(symbol, asset_ccy)
        attempts: List[Tuple[str, str]] = []
        found = False

        for yf_symbol in tickers_to_try:
            success, price = try_fetch_price(yf_symbol)
            if success and price is not None:
                prices[symbol] = price
                logger.info(f"Price for {symbol} ({yf_symbol}): {price}")
                attempts.append((yf_symbol, "ok"))
                found = True
                break
            else:
                attempts.append((yf_symbol, "fail"))
                logger.debug(f"No price data for {yf_symbol}")

        if not found:
            logger.warning(f"No price data for {symbol} after trying: {tickers_to_try}")
            lookup_attempts[symbol] = attempts

    return prices, lookup_attempts


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
) -> Tuple[Dict[str, Decimal], Decimal, List[Tuple[str, Decimal]]]:
    """
    Compute P&L for all positions.

    Returns:
        - user_pnl: dict of user_id -> total P&L in TWD
        - total_pnl: overall total P&L in TWD
        - symbol_pnl: list of (symbol, pnl_twd) sorted by absolute value
    """
    user_pnl: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    symbol_pnl: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
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

        logger.info(
            f"{user_id}/{symbol}/{asset_ccy}: "
            f"realized={result.realized_pnl}, unrealized={result.unrealized_pnl}, "
            f"pnl_twd={pnl_twd:.2f}"
        )

    sorted_symbols = sorted(
        symbol_pnl.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return dict(user_pnl), total_pnl, sorted_symbols


def format_slack_message(
    today: str,
    usd_twd: Decimal,
    total_pnl: Decimal,
    user_pnl: Dict[str, Decimal],
    top_symbols: List[Tuple[str, Decimal]],
    symbols_count: int,
    missing_symbols: List[str],
    lookup_attempts: Dict[str, List[Tuple[str, str]]],
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
    lines.append("*Top 5 Symbols by P&L (absolute):*")

    for i, (symbol, pnl) in enumerate(top5, 1):
        lines.append(f"  {i}. {symbol}: {pnl:+,.0f} TWD")

    lines.append("")
    lines.append(f"Top5 subtotal: {top5_subtotal:+,.0f} TWD")
    lines.append(f"Others subtotal: {others_subtotal:+,.0f} TWD")
    lines.append(f"Grand total: {total_pnl:+,.0f} TWD")

    lines.append("")
    missing_str = ", ".join(missing_symbols) if missing_symbols else "None"
    lines.append(f"Missing prices: {missing_str}")

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
    prices, lookup_attempts = get_close_prices(symbols_with_ccy)
    logger.info(f"Fetched prices for {len(prices)} symbols")

    all_symbols = {sym for sym, _ in symbols_with_ccy}
    missing_symbols = sorted(all_symbols - set(prices.keys()))

    grouped_txs = build_transactions(trades)

    user_pnl, total_pnl, top_symbols = compute_all_pnl(grouped_txs, prices, usd_twd)

    today = date.today().isoformat()
    message = format_slack_message(
        today, usd_twd, total_pnl, user_pnl, top_symbols,
        symbols_count=len(prices),
        missing_symbols=missing_symbols,
        lookup_attempts=lookup_attempts,
    )

    logger.info("Posting to Slack")
    post_to_slack(message)
    logger.info("Daily close completed successfully")


if __name__ == "__main__":
    main()
