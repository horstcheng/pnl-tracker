import json
import logging
import os
import urllib.request
from collections import defaultdict
from datetime import date, timedelta
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


def normalize_twd_ticker(symbol: str) -> str:
    """
    Normalize a TWD symbol to its yfinance ticker.

    Always treats input as string, preserving leading zeros and letters.
    Ensures ".TW" suffix for TWD assets.

    Args:
        symbol: The trading symbol (e.g., "00983A", "2330", "0050").

    Returns:
        The normalized yfinance ticker with .TW suffix.

    Examples:
        normalize_twd_ticker("00983A") -> "00983A.TW"
        normalize_twd_ticker("2330") -> "2330.TW"
        normalize_twd_ticker("0050") -> "0050.TW"
        normalize_twd_ticker("2330.TW") -> "2330.TW"  # already has suffix
    """
    # Always treat input as string, preserving leading zeros and letters
    symbol = str(symbol).strip()

    # If already has a suffix, return as-is
    if "." in symbol:
        return symbol

    # Append .TW suffix for TWD symbols
    return f"{symbol}.TW"


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
    asset_ccy = str(asset_ccy).strip().upper()

    # For TWD symbols, use normalize_twd_ticker for consistent handling
    if asset_ccy == "TWD":
        return normalize_twd_ticker(symbol)

    return symbol


def get_yfinance_tickers_to_try(symbol: str, asset_ccy: str) -> List[str]:
    """
    Return list of yfinance ticker symbols to try for a given symbol.

    Uses normalize_ticker() for the primary ticker, then adds .TWO fallback for TWD.
    """
    symbol = str(symbol).strip()
    asset_ccy = str(asset_ccy).strip().upper()
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
) -> Tuple[bool, Optional[Decimal], Optional[Decimal], Optional[dict]]:
    """
    Try to fetch current and previous trading day close prices.

    Primary method: Fetches 7 days of history and uses last two valid Close values.
    Fallback: When history has < 2 rows, tries fast_info["previousClose"].

    Args:
        yf_symbol: The yfinance ticker symbol to fetch.
        original_symbol: The original symbol (for debug logging).

    Returns (success, today_close, prev_close, debug_info) tuple.
    prev_close may be None even if success is True (insufficient history and no fallback).
    debug_info contains yf_ticker, rows, dates, closes, fallback info for debugging.
    """
    display_symbol = original_symbol or yf_symbol
    try:
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="7d")
        if hist.empty:
            return False, None, None, None
        close_series = hist["Close"].dropna()
        if close_series.empty:
            return False, None, None, None
        today_close = Decimal(str(close_series.iloc[-1]))

        # Capture debug info for day pnl (before deciding prev_close)
        num_rows = len(close_series)
        last_3_dates = [
            str(close_series.index[i].date())
            for i in range(-min(3, num_rows), 0)
        ] if num_rows >= 1 else []
        last_3_closes = [
            f"{float(close_series.iloc[i]):.2f}"
            for i in range(-min(3, num_rows), 0)
        ] if num_rows >= 1 else []

        debug_info = {
            "yf_ticker": yf_symbol,
            "rows": num_rows,
            "dates": last_3_dates,
            "closes": last_3_closes,
            "fallback_used": False,
            "fallback_prev_close": None,
        }

        prev_close = None
        if num_rows >= 2:
            # Primary method: use history
            prev_close = Decimal(str(close_series.iloc[-2]))
        else:
            # Fallback: try fast_info.previousClose when history has < 2 rows
            try:
                fast_info = ticker.fast_info
                fallback_prev = fast_info.get("previousClose") if fast_info else None
                if fallback_prev is not None:
                    prev_close = Decimal(str(fallback_prev))
                    debug_info["fallback_used"] = True
                    debug_info["fallback_prev_close"] = f"{float(fallback_prev):.2f}"
                    logger.info(
                        f"Day P&L fallback: symbol={display_symbol}, ticker={yf_symbol}, "
                        f"fast_info.previousClose={fallback_prev}"
                    )
            except Exception as fallback_err:
                logger.debug(f"Fallback fast_info failed for {yf_symbol}: {fallback_err}")

        logger.info(
            f"Day P&L debug: symbol={display_symbol}, ticker={yf_symbol}, "
            f"history_rows={num_rows}, last_3={list(zip(last_3_dates, last_3_closes))}, "
            f"fallback_used={debug_info['fallback_used']}"
        )

        return True, today_close, prev_close, debug_info
    except Exception as e:
        logger.debug(f"Failed to fetch {yf_symbol}: {e}")
        return False, None, None, None


def get_close_prices(
    symbols_with_ccy: List[Tuple[str, str]]
) -> Tuple[Dict[str, Decimal], Dict[str, Decimal], Dict[str, List[Tuple[str, str]]], List[str], Dict[str, dict], int, int]:
    """
    Fetch close prices for all symbols using yfinance.

    Returns:
        - prices: dict of symbol -> Decimal price (today's close)
        - prev_prices: dict of symbol -> Decimal price (previous trading day close)
        - lookup_attempts: dict of symbol -> list of (ticker, result) for missing symbols
        - missing_prev_close: list of symbols missing previous close data
        - missing_prev_debug: dict of symbol -> debug info for missing prev_close symbols
        - history_count: number of symbols using history-based prev_close
        - fallback_count: number of symbols using fast_info.previousClose fallback
    """
    prices: Dict[str, Decimal] = {}
    prev_prices: Dict[str, Decimal] = {}
    lookup_attempts: Dict[str, List[Tuple[str, str]]] = {}
    missing_prev_close: List[str] = []
    missing_prev_debug: Dict[str, dict] = {}
    history_count = 0
    fallback_count = 0

    for symbol, asset_ccy in symbols_with_ccy:
        tickers_to_try = get_yfinance_tickers_to_try(symbol, asset_ccy)
        attempts: List[Tuple[str, str]] = []
        found = False

        for yf_symbol in tickers_to_try:
            success, today_price, prev_price, debug_info = try_fetch_price_with_prev(yf_symbol, original_symbol=symbol)
            if success and today_price is not None:
                prices[symbol] = today_price
                if prev_price is not None:
                    prev_prices[symbol] = prev_price
                    # Track data source for prev_close
                    if debug_info and debug_info.get("fallback_used"):
                        fallback_count += 1
                    else:
                        history_count += 1
                else:
                    missing_prev_close.append(symbol)
                    if debug_info:
                        missing_prev_debug[symbol] = debug_info
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

    return prices, prev_prices, lookup_attempts, missing_prev_close, missing_prev_debug, history_count, fallback_count


# Sprint D-lite: Fetch historical prices only for Top-N symbols to minimize network calls.
# N=10 ensures enough symbols for top1/top3 trend computation even if some are missing.
TOP_N_FOR_BASELINE = 10


# Sprint D: Historical price fetch for risk trend (7D baseline)
def fetch_historical_prices(
    symbols_with_ccy: List[Tuple[str, str]],
    target_date: date,
) -> Tuple[Dict[str, Decimal], List[str]]:
    """
    Fetch close prices for symbols at or before a target date.

    Uses yfinance history with a date range ending at target_date + 1 day,
    then takes the last available close in the range. This handles weekends
    and holidays by using the nearest prior trading day.

    Args:
        symbols_with_ccy: list of (symbol, asset_ccy) tuples
        target_date: the target date (e.g., today - 7 days)

    Returns:
        - prices: dict of symbol -> Decimal close price
        - missing: list of symbols where price could not be fetched
    """
    prices: Dict[str, Decimal] = {}
    missing: List[str] = []

    # Fetch 10 calendar days ending at target_date to handle holidays/weekends
    start_date = target_date - timedelta(days=10)
    end_date = target_date + timedelta(days=1)  # yfinance end is exclusive

    for symbol, asset_ccy in symbols_with_ccy:
        tickers_to_try = get_yfinance_tickers_to_try(symbol, asset_ccy)
        found = False

        for yf_symbol in tickers_to_try:
            try:
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())
                if hist.empty:
                    continue
                close_series = hist["Close"].dropna()
                if close_series.empty:
                    continue
                
                # Require at least 2 data points to consider the history reliable/sufficient
                if len(close_series) < 2:
                    logger.debug(f"Insufficient historical data for {symbol} ({yf_symbol}): {len(close_series)} rows")
                    continue

                # Use last available close in the range (nearest prior trading day)
                close_price = Decimal(str(close_series.iloc[-1]))
                prices[symbol] = close_price
                found = True
                logger.debug(f"Historical price for {symbol} ({yf_symbol}) at {target_date}: {close_price}")
                break
            except Exception as e:
                logger.debug(f"Failed to fetch historical price for {yf_symbol}: {e}")
                continue

        if not found:
            missing.append(symbol)
            logger.debug(f"No historical price for {symbol} at {target_date}")

    return prices, missing


def get_top_n_symbols_with_ccy(
    concentration: List[Tuple[str, Decimal, Decimal]],
    symbol_positions: Dict[str, Tuple[Decimal, str]],
    n: int = TOP_N_FOR_BASELINE,
) -> List[Tuple[str, str]]:
    """
    Get top N symbols by market value with their currency classification.

    D-lite optimization: Instead of fetching historical prices for ALL symbols,
    fetch only for the top N symbols by today's market value. This minimizes
    network calls while preserving accuracy for concentration trend computation.

    Args:
        concentration: list of (symbol, pct, value_twd) sorted by pct descending
            (from compute_risk_views)
        symbol_positions: dict of symbol -> (quantity, asset_ccy)
        n: number of top symbols to return (default TOP_N_FOR_BASELINE)

    Returns:
        list of (symbol, asset_ccy) tuples for top N symbols
    """
    result = []
    for symbol, _pct, _value in concentration[:n]:
        if symbol in symbol_positions:
            _qty, asset_ccy = symbol_positions[symbol]
            result.append((symbol, asset_ccy))
    return result


def build_transactions(trades: List[dict]) -> Dict[Tuple[str, str, str], List[Transaction]]:
    """Group trades by (user_id, symbol, asset_ccy) and build Transaction objects."""
    grouped: Dict[Tuple[str, str, str], List[Transaction]] = defaultdict(list)

    for idx, trade in enumerate(trades, start=2):
        row_context = f"row {idx}"

        user_id = str(trade["user_id"]).strip()
        symbol = validate_symbol(trade["symbol"], row_context)
        asset_ccy = str(trade["asset_ccy"]).strip().upper()
        side = str(trade["side"]).strip().upper()
        
        # Validate numeric fields
        try:
            quantity = Decimal(str(trade["quantity"]))
            if quantity < 0:
                raise ValueError(f"Quantity must be non-negative, got {quantity} ({row_context})")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid quantity {trade.get('quantity')}: {e} ({row_context})")
        
        try:
            price = Decimal(str(trade["price"]))
            if price < 0:
                raise ValueError(f"Price must be non-negative, got {price} ({row_context})")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid price {trade.get('price')}: {e} ({row_context})")
        
        try:
            fee = Decimal(str(trade["fee"]))
            if fee < 0:
                raise ValueError(f"Fee must be non-negative, got {fee} ({row_context})")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid fee {trade.get('fee')}: {e} ({row_context})")
        
        trade_date = str(trade["trade_date"]).strip()
        if not trade_date:
            raise ValueError(f"Trade date cannot be empty ({row_context})")

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


# Sprint A frozen: Total P&L logic. Do not modify without CIO approval.
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
            if existing_ccy != asset_ccy:
                logger.warning(
                    f"Currency mismatch for {symbol}: existing {existing_ccy} vs {asset_ccy}"
                )
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


# Sprint C: Risk Views (Concentration Risk & Currency Exposure)
def compute_risk_views(
    symbol_positions: Dict[str, Tuple[Decimal, str]],
    prices: Dict[str, Decimal],
    usd_twd: Decimal,
) -> Tuple[
    Decimal,  # total_portfolio_value_twd
    List[Tuple[str, Decimal, Decimal]],  # concentration: (symbol, pct, value_twd) sorted desc by pct
    List[Tuple[str, Decimal, Decimal]],  # currency_exposure: (currency, pct, value_twd) sorted desc by pct
]:
    """
    Compute risk views: concentration risk and currency exposure.

    Uses already computed prices (today_close), symbol_positions (quantity, asset_ccy),
    and usd_twd. Does NOT fetch new prices or FX rates.

    market_value_twd(symbol) = quantity * today_close * fx_to_twd
    concentration_pct(symbol) = market_value_twd(symbol) / total_portfolio_value_twd
    currency_exposure_pct(ccy) = sum(market_value_twd for ccy) / total_portfolio_value_twd

    Args:
        symbol_positions: dict of symbol -> (quantity, asset_ccy)
        prices: dict of symbol -> today's close price
        usd_twd: USD/TWD exchange rate

    Returns:
        - total_portfolio_value_twd: sum of all market values in TWD
        - concentration: list of (symbol, concentration_pct, market_value_twd) sorted by pct desc
        - currency_exposure: list of (currency, exposure_pct, exposure_value_twd) sorted by pct desc
    """
    # Compute market_value_twd for each symbol
    symbol_market_values: Dict[str, Decimal] = {}
    currency_values: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

    for symbol, (quantity, asset_ccy) in symbol_positions.items():
        if symbol not in prices:
            # Skip symbols without price data
            continue
        if quantity == Decimal("0"):
            # No position, no market value
            continue

        today_close = prices[symbol]
        market_value_native = quantity * today_close

        # Convert to TWD
        if asset_ccy == "TWD":
            fx_to_twd = Decimal("1")
        elif asset_ccy == "USD":
            fx_to_twd = usd_twd
        else:
            # Unknown currency, treat as TWD (consistent with compute_all_pnl)
            fx_to_twd = Decimal("1")

        market_value_twd = market_value_native * fx_to_twd
        symbol_market_values[symbol] = market_value_twd
        currency_values[asset_ccy] += market_value_twd

    # Total portfolio value
    total_portfolio_value_twd = sum(symbol_market_values.values())

    # Concentration risk (by symbol)
    concentration: List[Tuple[str, Decimal, Decimal]] = []
    if total_portfolio_value_twd > Decimal("0"):
        for symbol, value_twd in symbol_market_values.items():
            pct = (value_twd / total_portfolio_value_twd) * Decimal("100")
            concentration.append((symbol, pct, value_twd))
    # Sort by pct descending
    concentration.sort(key=lambda x: x[1], reverse=True)

    # Currency exposure
    currency_exposure: List[Tuple[str, Decimal, Decimal]] = []
    if total_portfolio_value_twd > Decimal("0"):
        for ccy, value_twd in currency_values.items():
            pct = (value_twd / total_portfolio_value_twd) * Decimal("100")
            currency_exposure.append((ccy, pct, value_twd))
    # Sort by pct descending
    currency_exposure.sort(key=lambda x: x[1], reverse=True)

    return total_portfolio_value_twd, concentration, currency_exposure


# Pure helper functions for concentration and trend computation (no network, no side effects)
def compute_market_values_twd(
    positions: Dict[str, Tuple[Decimal, str]],
    prices: Dict[str, Decimal],
    usd_twd: Decimal,
) -> Dict[str, Decimal]:
    """
    Compute market values in TWD for each symbol.

    Pure function. Reuses the same currency/FX-to-TWD logic as Sprint C compute_risk_views.

    Args:
        positions: dict of symbol -> (quantity, asset_ccy)
        prices: dict of symbol -> close price
        usd_twd: USD/TWD exchange rate

    Returns:
        dict of symbol -> market_value_twd
    """
    result: Dict[str, Decimal] = {}

    for symbol, (quantity, asset_ccy) in positions.items():
        if symbol not in prices:
            continue
        if quantity == Decimal("0"):
            continue

        close_price = prices[symbol]
        market_value_native = quantity * close_price

        # Same FX logic as Sprint C compute_risk_views
        if asset_ccy == "TWD":
            fx_to_twd = Decimal("1")
        elif asset_ccy == "USD":
            fx_to_twd = usd_twd
        else:
            # Unknown currency, treat as TWD (consistent with compute_risk_views)
            fx_to_twd = Decimal("1")

        result[symbol] = market_value_native * fx_to_twd

    return result


def compute_concentration_from_values(
    values_twd: Dict[str, Decimal],
) -> List[Tuple[str, Decimal, Decimal]]:
    """
    Compute concentration list from market values in TWD.

    Pure function. Computes percentage of total for each symbol.

    Args:
        values_twd: dict of symbol -> market_value_twd

    Returns:
        list of (symbol, pct, value_twd) sorted by pct descending
    """
    total = sum(values_twd.values())

    concentration: List[Tuple[str, Decimal, Decimal]] = []
    if total > Decimal("0"):
        for symbol, value_twd in values_twd.items():
            pct = (value_twd / total) * Decimal("100")
            concentration.append((symbol, pct, value_twd))

    concentration.sort(key=lambda x: x[1], reverse=True)
    return concentration


def compute_topk_concentration_metrics(
    concentration: List[Tuple[str, Decimal, Decimal]],
    k: int = 3,
) -> Tuple[Decimal, Decimal]:
    """
    Compute Top-1 and Top-K concentration metrics from sorted concentration list.

    Pure function for testability.

    Args:
        concentration: list of (symbol, pct, value_twd) sorted by pct descending
        k: number of top symbols for sum metric (default 3)

    Returns:
        (top1_pct, topk_pct_sum)
    """
    if not concentration:
        return Decimal("0"), Decimal("0")

    top1_pct = concentration[0][1]
    topk_pct_sum = sum(c[1] for c in concentration[:k])
    return top1_pct, topk_pct_sum


def compute_trend_pp(
    base_pct: Decimal,
    today_pct: Decimal,
) -> Decimal:
    """
    Compute trend delta in percentage points.

    Pure function. Returns today - base (positive means increase).

    Args:
        base_pct: baseline percentage
        today_pct: today's percentage

    Returns:
        delta in percentage points (today - base)
    """
    return today_pct - base_pct


# Sprint D: Concentration trend helper functions
def compute_concentration_weights(
    concentration: List[Tuple[str, Decimal, Decimal]],
) -> Tuple[Decimal, Decimal]:
    """
    Compute Top-1 and Top-3 concentration weights from sorted concentration list.

    Pure function for testability.

    Args:
        concentration: list of (symbol, pct, value_twd) sorted by pct descending

    Returns:
        (top1_pct, top3_pct)
    """
    if not concentration:
        return Decimal("0"), Decimal("0")

    top1_pct = concentration[0][1]
    top3_pct = sum(c[1] for c in concentration[:3])
    return top1_pct, top3_pct


def compute_concentration_from_prices(
    symbol_positions: Dict[str, Tuple[Decimal, str]],
    prices: Dict[str, Decimal],
    usd_twd: Decimal,
) -> List[Tuple[str, Decimal, Decimal]]:
    """
    Compute concentration list from positions and prices.

    Reuses logic from compute_risk_views but returns only concentration.
    Used for computing baseline concentration with historical prices.

    Design note: Uses TODAY's usd_twd for both today and baseline.
    This is intentional for trend comparison (isolate price movement effect).

    Args:
        symbol_positions: dict of symbol -> (quantity, asset_ccy)
        prices: dict of symbol -> close price
        usd_twd: USD/TWD exchange rate (today's rate used for both periods)

    Returns:
        concentration: list of (symbol, pct, value_twd) sorted by pct descending
    """
    symbol_market_values: Dict[str, Decimal] = {}

    for symbol, (quantity, asset_ccy) in symbol_positions.items():
        if symbol not in prices:
            continue
        if quantity == Decimal("0"):
            continue

        close_price = prices[symbol]
        market_value_native = quantity * close_price

        if asset_ccy == "TWD":
            fx_to_twd = Decimal("1")
        elif asset_ccy == "USD":
            fx_to_twd = usd_twd
        else:
            fx_to_twd = Decimal("1")

        market_value_twd = market_value_native * fx_to_twd
        symbol_market_values[symbol] = market_value_twd

    total_portfolio_value_twd = sum(symbol_market_values.values())

    concentration: List[Tuple[str, Decimal, Decimal]] = []
    if total_portfolio_value_twd > Decimal("0"):
        for symbol, value_twd in symbol_market_values.items():
            pct = (value_twd / total_portfolio_value_twd) * Decimal("100")
            concentration.append((symbol, pct, value_twd))

    concentration.sort(key=lambda x: x[1], reverse=True)
    return concentration


def compute_concentration_trend(
    symbol_positions: Dict[str, Tuple[Decimal, str]],
    today_prices: Dict[str, Decimal],
    baseline_prices: Dict[str, Decimal],
    usd_twd: Decimal,
    min_symbols_required: int = 3,
) -> Optional[Dict[str, Decimal]]:
    """
    Compute concentration trend: today vs 7D-ago baseline.

    Returns trend data if at least min_symbols_required have both today and baseline prices.
    Returns None if insufficient data (graceful degradation).

    Design note: Uses TODAY's usd_twd for both periods to isolate price movement effect.

    Args:
        symbol_positions: dict of symbol -> (quantity, asset_ccy)
        today_prices: dict of symbol -> today's close price
        baseline_prices: dict of symbol -> 7D-ago close price
        usd_twd: TODAY's USD/TWD rate (used for both periods)
        min_symbols_required: minimum symbols needed to compute trend (default 3)

    Returns:
        dict with keys: baseline_top1, today_top1, delta_top1,
                        baseline_top3, today_top3, delta_top3
        Or None if insufficient data.
    """
    # Count symbols with both prices
    symbols_with_both = [
        sym for sym in symbol_positions
        if sym in today_prices and sym in baseline_prices
    ]

    if len(symbols_with_both) < min_symbols_required:
        logger.warning(
            f"Concentration trend: insufficient data ({len(symbols_with_both)} symbols, "
            f"need {min_symbols_required})"
        )
        return None

    # Compute today's concentration
    today_concentration = compute_concentration_from_prices(
        symbol_positions, today_prices, usd_twd
    )
    today_top1, today_top3 = compute_concentration_weights(today_concentration)

    # Compute baseline concentration (7D ago prices, today's FX)
    baseline_concentration = compute_concentration_from_prices(
        symbol_positions, baseline_prices, usd_twd
    )
    baseline_top1, baseline_top3 = compute_concentration_weights(baseline_concentration)

    return {
        "baseline_top1": baseline_top1,
        "today_top1": today_top1,
        "delta_top1": today_top1 - baseline_top1,
        "baseline_top3": baseline_top3,
        "today_top3": today_top3,
        "delta_top3": today_top3 - baseline_top3,
    }


def compute_currency_exposure_from_prices(
    symbol_positions: Dict[str, Tuple[Decimal, str]],
    prices: Dict[str, Decimal],
    usd_twd: Decimal,
) -> List[Tuple[str, Decimal, Decimal]]:
    """
    Compute currency exposure from positions and prices.

    Groups market values by currency and computes percentage exposure.
    Used for computing baseline currency exposure with historical prices.

    Design note: Uses TODAY's usd_twd for both today and baseline.
    This is intentional for trend comparison (isolate price movement effect).

    Args:
        symbol_positions: dict of symbol -> (quantity, asset_ccy)
        prices: dict of symbol -> close price
        usd_twd: USD/TWD exchange rate (today's rate used for both periods)

    Returns:
        currency_exposure: list of (ccy, pct, value_twd) sorted by pct descending
    """
    currency_values: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

    for symbol, (quantity, asset_ccy) in symbol_positions.items():
        if symbol not in prices:
            continue
        if quantity == Decimal("0"):
            continue

        close_price = prices[symbol]
        market_value_native = quantity * close_price

        if asset_ccy == "TWD":
            fx_to_twd = Decimal("1")
        elif asset_ccy == "USD":
            fx_to_twd = usd_twd
        else:
            fx_to_twd = Decimal("1")

        market_value_twd = market_value_native * fx_to_twd
        currency_values[asset_ccy] += market_value_twd

    total_portfolio_value_twd = sum(currency_values.values())

    currency_exposure: List[Tuple[str, Decimal, Decimal]] = []
    if total_portfolio_value_twd > Decimal("0"):
        for ccy, value_twd in currency_values.items():
            pct = (value_twd / total_portfolio_value_twd) * Decimal("100")
            currency_exposure.append((ccy, pct, value_twd))

    currency_exposure.sort(key=lambda x: x[1], reverse=True)
    return currency_exposure


def compute_currency_exposure_trend(
    symbol_positions: Dict[str, Tuple[Decimal, str]],
    today_prices: Dict[str, Decimal],
    baseline_prices: Dict[str, Decimal],
    usd_twd: Decimal,
    min_symbols_required: int = 3,
) -> Optional[List[Dict[str, Decimal]]]:
    """
    Compute currency exposure trend: today vs 7D-ago baseline.

    Returns trend data if at least min_symbols_required have both today and baseline prices.
    Returns None if insufficient data (graceful degradation).

    Design note: Uses TODAY's usd_twd for both periods to isolate price movement effect.

    Args:
        symbol_positions: dict of symbol -> (quantity, asset_ccy)
        today_prices: dict of symbol -> today's close price
        baseline_prices: dict of symbol -> 7D-ago close price
        usd_twd: TODAY's USD/TWD rate (used for both periods)
        min_symbols_required: minimum symbols needed to compute trend (default 3)

    Returns:
        list of dicts, each with keys: ccy, baseline_pct, today_pct, delta_pct
        Sorted by today_pct descending. Includes all currencies in union of baseline and today.
        Or None if insufficient data.
    """
    # Count symbols with both prices
    symbols_with_both = [
        sym for sym in symbol_positions
        if sym in today_prices and sym in baseline_prices
    ]

    if len(symbols_with_both) < min_symbols_required:
        logger.warning(
            f"Currency exposure trend: insufficient data ({len(symbols_with_both)} symbols, "
            f"need {min_symbols_required})"
        )
        return None

    # Compute today's currency exposure
    today_exposure = compute_currency_exposure_from_prices(
        symbol_positions, today_prices, usd_twd
    )
    today_by_ccy = {ccy: pct for ccy, pct, _ in today_exposure}

    # Compute baseline currency exposure (7D ago prices, today's FX)
    baseline_exposure = compute_currency_exposure_from_prices(
        symbol_positions, baseline_prices, usd_twd
    )
    baseline_by_ccy = {ccy: pct for ccy, pct, _ in baseline_exposure}

    # Union of all currencies
    all_currencies = set(today_by_ccy.keys()) | set(baseline_by_ccy.keys())

    result = []
    for ccy in all_currencies:
        today_pct = today_by_ccy.get(ccy, Decimal("0"))
        baseline_pct = baseline_by_ccy.get(ccy, Decimal("0"))
        result.append({
            "ccy": ccy,
            "baseline_pct": baseline_pct,
            "today_pct": today_pct,
            "delta_pct": today_pct - baseline_pct,
        })

    # Sort by today's exposure descending
    result.sort(key=lambda x: x["today_pct"], reverse=True)
    return result


# Sprint A frozen: Day P&L logic (history + previousClose fallback).
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
            logger.warning(f"Unknown currency {asset_ccy} for {symbol}, treating as TWD")
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


# Sprint E: Symbol display names (zh-TW) for Slack report readability
# Display-only mapping. Does NOT affect symbol keys used for calculations or price lookup.
SYMBOL_NAMES_ZH = {
    "0050": "元大台灣50",
    "0052": "富邦科技",
    "00687B": "國泰20年美債",
    "00713": "元大台灣高息低波",
    "00953B": "國泰10Y+金融債",
    "00965": "富邦A級公司債",
    "00972": "凱基美國非投等債",
    "00983A": "中信美10Y+A公司債",
    "00984A": "中信10Y+金融債",
    "00988A": "凱基AAA-A公司債",
    "009805": "元大20年美債",
    "009812": "元大投等債15Y+",
    "1519": "華城",
    "4979": "華星光",
    "6442": "光聖",
    "6789": "采鈺",
    "2330": "台積電",
}


def display_symbol(symbol: str) -> str:
    """
    Return display string for symbol: 'SYMBOL（NAME）' if mapped, else 'SYMBOL'.

    Display-only. Does NOT affect underlying symbol keys used for calculations.
    """
    name = SYMBOL_NAMES_ZH.get(symbol)
    if name:
        return f"{symbol}（{name}）"
    return symbol


# zh-TW display labels for Slack report (display-only, no logic change)
LABELS_ZH = {
    "daily_report": "每日損益報表",
    "usd_twd": "USD/TWD",
    "symbols_counted": "計入標的數",
    "user_ranking": "用戶損益排名（前3名）",
    "top5_total": "前5大標的（累計）",
    "top5_subtotal": "前5大小計",
    "others_subtotal": "其他小計",
    "grand_total": "總計",
    "total_today": "今日損益",
    "top5_today": "前5大標的（今日）",
    "day_pnl_sources": "今日損益資料來源",
    "history": "歷史資料",
    "fallback": "previousClose備援",
    "symbols_unit": "檔",
    "day_pnl_warnings": "今日損益警告",
    "missing_prev_close": "缺少前日收盤價",
    "day_pnl_note": "今日損益備註：缺少前日收盤價之標的以0計算（不列入今日變動）",
    "day_pnl_debug": "今日損益除錯",
    "missing_prices": "缺少報價（累計）",
    "none": "無",
    "lookup_attempts": "報價查詢嘗試（僅缺少者）",
    "risk_views": "風險視圖",
    "concentration_risk": "集中度風險（依市值）",
    "top1_alert": "單一標的集中度風險",
    "top3_alert": "前三大集中度風險",
    "currency_exposure": "幣別曝險",
    # Sprint D-lite: Risk trend labels (Top-10 baseline)
    "risk_trend_7d": "風險趨勢（7日｜Top-10）",
    "top1_concentration": "Top-1 集中度",
    "top3_concentration": "Top-3 集中度",
    "risk_trend_unavailable": "無法取得足夠的歷史價格，略過。",
    "currency_exposure_trend_7d": "幣別曝險趨勢（7日）",
}


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
    missing_prev_debug: Optional[Dict[str, dict]] = None,
    day_pnl_history_count: int = 0,
    day_pnl_fallback_count: int = 0,
    concentration: Optional[List[Tuple[str, Decimal, Decimal]]] = None,
    currency_exposure: Optional[List[Tuple[str, Decimal, Decimal]]] = None,
    concentration_trend: Optional[Dict[str, Decimal]] = None,
    currency_exposure_trend: Optional[List[Dict[str, Decimal]]] = None,
) -> str:
    """Format the Slack message."""
    L = LABELS_ZH
    top5 = top_symbols[:5]
    top5_subtotal = sum(pnl for _, pnl in top5)
    others_subtotal = total_pnl - top5_subtotal

    lines = [
        f"*{L['daily_report']} - {today}*",
        "",
        f"{L['usd_twd']}：{usd_twd:.2f}",
        f"{L['symbols_counted']}：{symbols_count}",
        "",
        f"*{L['user_ranking']}：*",
    ]

    sorted_users = sorted(user_pnl.items(), key=lambda x: x[1], reverse=True)
    for i, (user_id, pnl) in enumerate(sorted_users[:3], 1):
        lines.append(f"  {i}. {user_id}：{pnl:+,.0f} TWD")

    lines.append("")
    lines.append(f"*{L['top5_total']}：*")

    for i, (symbol, pnl) in enumerate(top5, 1):
        lines.append(f"  {i}. {display_symbol(symbol)}：{pnl:+,.0f} TWD")

    lines.append("")
    lines.append(f"{L['top5_subtotal']}：{top5_subtotal:+,.0f} TWD")
    lines.append(f"{L['others_subtotal']}：{others_subtotal:+,.0f} TWD")
    lines.append(f"{L['grand_total']}：{total_pnl:+,.0f} TWD")

    # Day P&L section (Today)
    lines.append("")
    lines.append(f"*{L['total_today']}：{total_day_pnl:+,.0f} TWD*")

    lines.append("")
    lines.append(f"*{L['top5_today']}：*")
    top5_day = top_day_symbols[:5]
    for i, (symbol, pnl) in enumerate(top5_day, 1):
        lines.append(f"  {i}. {display_symbol(symbol)}：{pnl:+,.0f} TWD")

    # Day P&L data source summary
    lines.append("")
    lines.append(f"{L['day_pnl_sources']}：{L['history']}={day_pnl_history_count}{L['symbols_unit']}，{L['fallback']}={day_pnl_fallback_count}{L['symbols_unit']}")

    # Day P&L warnings
    if missing_prev_close:
        lines.append("")
        lines.append(f"{L['day_pnl_warnings']}：{L['missing_prev_close']} {', '.join(sorted(missing_prev_close))}")
        lines.append(L['day_pnl_note'])

        # Day P&L debug section for missing prev_close symbols
        if missing_prev_debug:
            lines.append("")
            lines.append(f"{L['day_pnl_debug']}：")
            for sym in sorted(missing_prev_close):
                if sym in missing_prev_debug:
                    info = missing_prev_debug[sym]
                    yf_ticker = info.get("yf_ticker", "?")
                    rows = info.get("rows", "?")
                    closes = ", ".join(info.get("closes", []))
                    dates = ", ".join(info.get("dates", []))
                    fallback_used = info.get("fallback_used", False)
                    fallback_prev = info.get("fallback_prev_close", None)
                    fallback_str = f", fallback_used={fallback_used}, fallback_prev={fallback_prev}" if fallback_used or rows == 1 else ""
                    lines.append(f"- {sym}: yf_ticker={yf_ticker}, rows={rows}, closes={closes}, dates={dates}{fallback_str}")

    lines.append("")
    missing_str = ", ".join(missing_symbols) if missing_symbols else L['none']
    lines.append(f"{L['missing_prices']}：{missing_str}")

    if lookup_attempts:
        attempts_parts = []
        for sym in sorted(lookup_attempts.keys()):
            attempts = lookup_attempts[sym]
            attempt_str = ", ".join(f"{t}={r}" for t, r in attempts)
            attempts_parts.append(f"{sym}: {attempt_str}")
        lines.append(f"{L['lookup_attempts']}：{'; '.join(attempts_parts)}")

    # Sprint C: Risk Views section (append-only)
    if concentration is not None and currency_exposure is not None:
        lines.append("")
        lines.append(f"*{L['risk_views']}：*")

        # Concentration Risk (by market value)
        lines.append(f"*{L['concentration_risk']}：*")
        top5_concentration = concentration[:5]
        for i, (symbol, pct, value_twd) in enumerate(top5_concentration, 1):
            lines.append(f"  {i}. {display_symbol(symbol)}：{pct:.1f}%（{value_twd:,.0f}）")

        # Concentration risk alerts
        if concentration:
            top1_pct = concentration[0][1] if len(concentration) >= 1 else Decimal("0")
            top3_pct = sum(c[1] for c in concentration[:3])
            if top1_pct > Decimal("25"):
                lines.append(f"⚠️ {L['top1_alert']}")
            if top3_pct > Decimal("60"):
                lines.append(f"⚠️ {L['top3_alert']}")

        # Currency Exposure
        lines.append("")
        lines.append(f"*{L['currency_exposure']}：*")
        for ccy, pct, value_twd in currency_exposure:
            lines.append(f"  - {ccy}：{pct:.1f}%（{value_twd:,.0f}）")

    # Sprint D: Risk Trend (7D) section (append-only, after Risk Views)
    lines.append("")
    if concentration_trend is not None:
        lines.append(f"*{L['risk_trend_7d']}：*")
        lines.append(
            f"  - {L['top1_concentration']}："
            f"{concentration_trend['baseline_top1']:.1f}% → "
            f"{concentration_trend['today_top1']:.1f}%"
            f"（{concentration_trend['delta_top1']:+.1f}%）"
        )
        lines.append(
            f"  - {L['top3_concentration']}："
            f"{concentration_trend['baseline_top3']:.1f}% → "
            f"{concentration_trend['today_top3']:.1f}%"
            f"（{concentration_trend['delta_top3']:+.1f}%）"
        )
    else:
        lines.append(f"*{L['risk_trend_7d']}：*{L['risk_trend_unavailable']}")

    # Sprint D extension: Currency Exposure Trend (7D) section (append-only)
    lines.append("")
    if currency_exposure_trend is not None:
        lines.append(f"*{L['currency_exposure_trend_7d']}：*")
        for item in currency_exposure_trend:
            lines.append(
                f"  - {item['ccy']}："
                f"{item['baseline_pct']:.1f}% → "
                f"{item['today_pct']:.1f}%"
                f"（{item['delta_pct']:+.1f}%）"
            )
    else:
        lines.append(f"*{L['currency_exposure_trend_7d']}：*{L['risk_trend_unavailable']}")

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
    try:
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
        prices, prev_prices, lookup_attempts, missing_prev_close, missing_prev_debug, history_count, fallback_count = get_close_prices(symbols_with_ccy)
        logger.info(f"Fetched prices for {len(prices)} symbols")

        all_symbols = {sym for sym, _ in symbols_with_ccy}
        missing_symbols = sorted(all_symbols - set(prices.keys()))

        grouped_txs = build_transactions(trades)

        user_pnl, total_pnl, top_symbols, symbol_positions = compute_all_pnl(grouped_txs, prices, usd_twd)

        # Compute Day P&L
        total_day_pnl, top_day_symbols = compute_day_pnl(symbol_positions, prices, prev_prices, usd_twd)
        logger.info(f"Day P&L calculated: {total_day_pnl:.2f} TWD")

        # Compute Risk Views (Sprint C)
        total_portfolio_value, concentration, currency_exposure = compute_risk_views(
            symbol_positions, prices, usd_twd
        )
        logger.info(f"Risk views calculated: portfolio={total_portfolio_value:.0f} TWD, {len(concentration)} symbols, {len(currency_exposure)} currencies")

        # D-lite: Compute Concentration Trend (7D) using Top-10 symbols only
        # Fetch historical prices only for top 10 symbols by market value
        # to minimize network calls. Uses SAME fx_to_twd and currency classification as today.
        today_date = date.today()
        baseline_date = today_date - timedelta(days=7)
        top_10_symbols_with_ccy = get_top_n_symbols_with_ccy(concentration, symbol_positions, n=10)
        baseline_prices, baseline_missing = fetch_historical_prices(top_10_symbols_with_ccy, baseline_date)
        logger.info(
            f"D-lite: Fetched baseline prices for {len(baseline_prices)}/{len(top_10_symbols_with_ccy)} "
            f"top symbols (7D ago: {baseline_date}), missing: {len(baseline_missing)}"
        )

        # Compute baseline market values using baseline closes and the SAME FX-to-TWD logic as today
        baseline_market_values = compute_market_values_twd(symbol_positions, baseline_prices, usd_twd)
        baseline_concentration = compute_concentration_from_values(baseline_market_values)

        # Compute baseline concentration weights (Top-10 subset)
        if len(baseline_concentration) >= 3:
            baseline_top1, baseline_top3 = compute_concentration_weights(baseline_concentration)
            logger.info(
                f"Baseline concentration: Top-1 {baseline_top1:.1f}%, Top-3 {baseline_top3:.1f}%"
            )
        else:
            logger.warning("Baseline concentration unavailable (insufficient historical data)")

        # Compute concentration trend (uses today's usd_twd for both periods)
        concentration_trend = compute_concentration_trend(
            symbol_positions, prices, baseline_prices, usd_twd
        )
        if concentration_trend:
            logger.info(
                f"Concentration trend: Top-1 {concentration_trend['baseline_top1']:.1f}% -> {concentration_trend['today_top1']:.1f}%, "
                f"Top-3 {concentration_trend['baseline_top3']:.1f}% -> {concentration_trend['today_top3']:.1f}%"
            )
        else:
            logger.warning("Concentration trend unavailable (insufficient historical data)")

        # Compute currency exposure trend (uses today's usd_twd for both periods)
        currency_exposure_trend = compute_currency_exposure_trend(
            symbol_positions, prices, baseline_prices, usd_twd
        )
        if currency_exposure_trend:
            trend_summary = ", ".join(
                f"{item['ccy']} {item['baseline_pct']:.1f}%->{item['today_pct']:.1f}%"
                for item in currency_exposure_trend
            )
            logger.info(f"Currency exposure trend: {trend_summary}")
        else:
            logger.warning("Currency exposure trend unavailable (insufficient historical data)")

        today = today_date.isoformat()
        message = format_slack_message(
            today, usd_twd, total_pnl, user_pnl, top_symbols,
            symbols_count=len(prices),
            missing_symbols=missing_symbols,
            lookup_attempts=lookup_attempts,
            total_day_pnl=total_day_pnl,
            top_day_symbols=top_day_symbols,
            missing_prev_close=missing_prev_close,
            missing_prev_debug=missing_prev_debug,
            day_pnl_history_count=history_count,
            day_pnl_fallback_count=fallback_count,
            concentration=concentration,
            currency_exposure=currency_exposure,
            concentration_trend=concentration_trend,
            currency_exposure_trend=currency_exposure_trend,
        )

        logger.info("Posting to Slack")
        post_to_slack(message)
        logger.info("Daily close completed successfully")
    except Exception:
        logger.exception("An unhandled exception occurred during the daily close process.")
        raise


if __name__ == "__main__":
    main()
