# packages/pnl/tests/test_risk_trend.py
"""Unit tests for Sprint D Concentration Risk Trend (7D, no snapshot)."""
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from services.api.daily_close import (
    TOP_N_FOR_BASELINE,
    compute_concentration_weights,
    compute_concentration_from_prices,
    compute_concentration_trend,
    compute_currency_exposure_from_prices,
    compute_currency_exposure_trend,
    fetch_historical_prices,
    format_slack_message,
    get_top_n_symbols_with_ccy,
)


class TestComputeConcentrationWeights:
    """Tests for compute_concentration_weights pure function."""

    def test_basic_weights(self):
        """Test basic Top-1 and Top-3 weight computation."""
        concentration = [
            ("A", Decimal("40"), Decimal("400000")),
            ("B", Decimal("30"), Decimal("300000")),
            ("C", Decimal("20"), Decimal("200000")),
            ("D", Decimal("10"), Decimal("100000")),
        ]
        top1, top3 = compute_concentration_weights(concentration)
        assert top1 == Decimal("40")
        assert top3 == Decimal("90")

    def test_empty_concentration(self):
        """Test with empty concentration list."""
        top1, top3 = compute_concentration_weights([])
        assert top1 == Decimal("0")
        assert top3 == Decimal("0")

    def test_single_symbol(self):
        """Test with single symbol (Top-1 == Top-3)."""
        concentration = [("A", Decimal("100"), Decimal("1000000"))]
        top1, top3 = compute_concentration_weights(concentration)
        assert top1 == Decimal("100")
        assert top3 == Decimal("100")

    def test_two_symbols(self):
        """Test with two symbols (Top-3 is sum of both)."""
        concentration = [
            ("A", Decimal("60"), Decimal("600000")),
            ("B", Decimal("40"), Decimal("400000")),
        ]
        top1, top3 = compute_concentration_weights(concentration)
        assert top1 == Decimal("60")
        assert top3 == Decimal("100")


class TestComputeConcentrationFromPrices:
    """Tests for compute_concentration_from_prices pure function."""

    def test_basic_concentration(self):
        """Test concentration computation with mixed currencies."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),  # 100 * 150 * 32 = 480000 TWD
            "2330": (Decimal("1000"), "TWD"),  # 1000 * 600 = 600000 TWD
        }
        prices = {
            "AAPL": Decimal("150"),
            "2330": Decimal("600"),
        }
        usd_twd = Decimal("32")

        concentration = compute_concentration_from_prices(
            symbol_positions, prices, usd_twd
        )

        # Total: 480000 + 600000 = 1080000
        # 2330: 600000 / 1080000 = 55.56%
        # AAPL: 480000 / 1080000 = 44.44%
        assert len(concentration) == 2
        assert concentration[0][0] == "2330"  # Largest first
        assert abs(concentration[0][1] - Decimal("55.56")) < Decimal("0.01")
        assert concentration[1][0] == "AAPL"
        assert abs(concentration[1][1] - Decimal("44.44")) < Decimal("0.01")

    def test_skips_missing_prices(self):
        """Test that symbols without prices are skipped."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "UNKNOWN": (Decimal("50"), "USD"),
        }
        prices = {"AAPL": Decimal("150")}
        usd_twd = Decimal("32")

        concentration = compute_concentration_from_prices(
            symbol_positions, prices, usd_twd
        )

        assert len(concentration) == 1
        assert concentration[0][0] == "AAPL"
        assert concentration[0][1] == Decimal("100")

    def test_skips_zero_quantity(self):
        """Test that zero quantity symbols are skipped."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "SOLD": (Decimal("0"), "USD"),
        }
        prices = {
            "AAPL": Decimal("150"),
            "SOLD": Decimal("100"),
        }
        usd_twd = Decimal("32")

        concentration = compute_concentration_from_prices(
            symbol_positions, prices, usd_twd
        )

        assert len(concentration) == 1
        assert concentration[0][0] == "AAPL"


class TestComputeConcentrationTrend:
    """Tests for compute_concentration_trend function."""

    def test_trend_computation(self):
        """Test trend delta computation with synthetic data."""
        symbol_positions = {
            "A": (Decimal("100"), "TWD"),
            "B": (Decimal("100"), "TWD"),
            "C": (Decimal("100"), "TWD"),
        }
        # Today: A=50, B=30, C=20 -> Top1=50%, Top3=100%
        today_prices = {
            "A": Decimal("50"),
            "B": Decimal("30"),
            "C": Decimal("20"),
        }
        # Baseline: A=40, B=35, C=25 -> Top1=40%, Top3=100%
        baseline_prices = {
            "A": Decimal("40"),
            "B": Decimal("35"),
            "C": Decimal("25"),
        }
        usd_twd = Decimal("32")

        trend = compute_concentration_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd
        )

        assert trend is not None
        # Baseline: A=4000, B=3500, C=2500, Total=10000
        # A: 40%, B: 35%, C: 25% -> Top1=40%, Top3=100%
        assert trend["baseline_top1"] == Decimal("40")
        assert trend["baseline_top3"] == Decimal("100")

        # Today: A=5000, B=3000, C=2000, Total=10000
        # A: 50%, B: 30%, C: 20% -> Top1=50%, Top3=100%
        assert trend["today_top1"] == Decimal("50")
        assert trend["today_top3"] == Decimal("100")

        # Delta
        assert trend["delta_top1"] == Decimal("10")
        assert trend["delta_top3"] == Decimal("0")

    def test_trend_with_negative_delta(self):
        """Test trend with concentration decreasing (negative delta)."""
        symbol_positions = {
            "A": (Decimal("100"), "TWD"),
            "B": (Decimal("100"), "TWD"),
            "C": (Decimal("100"), "TWD"),
        }
        # Today: prices equal -> A=33.33%, B=33.33%, C=33.33%
        today_prices = {
            "A": Decimal("100"),
            "B": Decimal("100"),
            "C": Decimal("100"),
        }
        # Baseline: A dominant -> A=60%, B=C=20%
        baseline_prices = {
            "A": Decimal("150"),
            "B": Decimal("50"),
            "C": Decimal("50"),
        }
        usd_twd = Decimal("32")

        trend = compute_concentration_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd
        )

        assert trend is not None
        assert trend["baseline_top1"] == Decimal("60")
        assert abs(trend["today_top1"] - Decimal("33.33")) < Decimal("0.01")
        assert trend["delta_top1"] < Decimal("0")  # Negative delta

    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None gracefully."""
        symbol_positions = {
            "A": (Decimal("100"), "TWD"),
            "B": (Decimal("100"), "TWD"),
            "C": (Decimal("100"), "TWD"),
        }
        today_prices = {
            "A": Decimal("100"),
            "B": Decimal("100"),
            "C": Decimal("100"),
        }
        # Only 2 symbols have baseline prices (need 3)
        baseline_prices = {
            "A": Decimal("100"),
            "B": Decimal("100"),
        }
        usd_twd = Decimal("32")

        trend = compute_concentration_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd
        )

        assert trend is None

    def test_custom_min_symbols_required(self):
        """Test custom min_symbols_required parameter."""
        symbol_positions = {
            "A": (Decimal("100"), "TWD"),
            "B": (Decimal("100"), "TWD"),
        }
        today_prices = {
            "A": Decimal("100"),
            "B": Decimal("100"),
        }
        baseline_prices = {
            "A": Decimal("100"),
            "B": Decimal("100"),
        }
        usd_twd = Decimal("32")

        # Default min=3 should fail
        trend = compute_concentration_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd
        )
        assert trend is None

        # min=2 should succeed
        trend = compute_concentration_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd,
            min_symbols_required=2
        )
        assert trend is not None


class TestFetchHistoricalPrices:
    """Tests for fetch_historical_prices with mocked yfinance."""

    @patch("services.api.daily_close.yf.Ticker")
    def test_fetch_historical_prices_success(self, mock_ticker_class):
        """Test successful historical price fetch."""
        import pandas as pd

        # Setup mock
        mock_ticker = MagicMock()
        mock_hist = pd.DataFrame({
            "Close": [148.0, 149.0, 150.0],
        }, index=pd.to_datetime(["2026-01-10", "2026-01-11", "2026-01-12"]))
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker

        symbols_with_ccy = [("AAPL", "USD")]
        target_date = date(2026, 1, 13)

        prices, missing = fetch_historical_prices(symbols_with_ccy, target_date)

        assert "AAPL" in prices
        assert prices["AAPL"] == Decimal("150")
        assert missing == []

    @patch("services.api.daily_close.yf.Ticker")
    def test_fetch_historical_prices_missing(self, mock_ticker_class):
        """Test handling of missing historical prices."""
        import pandas as pd

        # Setup mock to return empty history
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        symbols_with_ccy = [("UNKNOWN", "USD")]
        target_date = date(2026, 1, 13)

        prices, missing = fetch_historical_prices(symbols_with_ccy, target_date)

        assert prices == {}
        assert "UNKNOWN" in missing

    @patch("services.api.daily_close.yf.Ticker")
    def test_fetch_historical_prices_handles_nan(self, mock_ticker_class):
        """Test that NaN values in close series are handled."""
        import pandas as pd
        import numpy as np

        mock_ticker = MagicMock()
        mock_hist = pd.DataFrame({
            "Close": [148.0, np.nan, 150.0],
        }, index=pd.to_datetime(["2026-01-10", "2026-01-11", "2026-01-12"]))
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker

        symbols_with_ccy = [("AAPL", "USD")]
        target_date = date(2026, 1, 13)

        prices, missing = fetch_historical_prices(symbols_with_ccy, target_date)

        # Should use last valid close after dropna
        assert "AAPL" in prices
        assert prices["AAPL"] == Decimal("150")


class TestFormatSlackMessageRiskTrend:
    """Tests for Risk Trend section in format_slack_message."""

    def _create_base_message_params(self):
        """Create base parameters for format_slack_message."""
        return {
            "today": "2026-01-20",
            "usd_twd": Decimal("32.5"),
            "total_pnl": Decimal("100000"),
            "user_pnl": {"user1": Decimal("100000")},
            "top_symbols": [("AAPL", Decimal("50000"))],
            "symbols_count": 5,
            "missing_symbols": [],
            "lookup_attempts": {},
            "total_day_pnl": Decimal("5000"),
            "top_day_symbols": [("AAPL", Decimal("3000"))],
            "missing_prev_close": [],
            "concentration": [("AAPL", Decimal("50"), Decimal("500000"))],
            "currency_exposure": [("USD", Decimal("100"), Decimal("500000"))],
        }

    def test_risk_trend_section_present(self):
        """Test that risk trend section is present when data available."""
        params = self._create_base_message_params()
        params["concentration_trend"] = {
            "baseline_top1": Decimal("40.5"),
            "today_top1": Decimal("45.2"),
            "delta_top1": Decimal("4.7"),
            "baseline_top3": Decimal("75.0"),
            "today_top3": Decimal("78.5"),
            "delta_top3": Decimal("3.5"),
        }

        message = format_slack_message(**params)

        assert "*風險趨勢（7日）：*" in message
        assert "Top-1 集中度" in message
        assert "Top-3 集中度" in message
        assert "40.5% → 45.2%" in message
        assert "+4.7%" in message
        assert "75.0% → 78.5%" in message
        assert "+3.5%" in message

    def test_risk_trend_negative_delta(self):
        """Test formatting of negative trend delta."""
        params = self._create_base_message_params()
        params["concentration_trend"] = {
            "baseline_top1": Decimal("50.0"),
            "today_top1": Decimal("45.0"),
            "delta_top1": Decimal("-5.0"),
            "baseline_top3": Decimal("80.0"),
            "today_top3": Decimal("75.0"),
            "delta_top3": Decimal("-5.0"),
        }

        message = format_slack_message(**params)

        assert "-5.0%" in message

    def test_risk_trend_unavailable(self):
        """Test fallback message when trend data unavailable."""
        params = self._create_base_message_params()
        params["concentration_trend"] = None

        message = format_slack_message(**params)

        assert "*風險趨勢（7日）：*" in message
        assert "無法取得足夠的歷史價格，略過。" in message

    def test_risk_trend_appended_after_risk_views(self):
        """Test that Risk Trend appears after Risk Views section."""
        params = self._create_base_message_params()
        params["concentration_trend"] = {
            "baseline_top1": Decimal("40"),
            "today_top1": Decimal("45"),
            "delta_top1": Decimal("5"),
            "baseline_top3": Decimal("75"),
            "today_top3": Decimal("80"),
            "delta_top3": Decimal("5"),
        }

        message = format_slack_message(**params)

        risk_views_pos = message.find("*風險視圖：*")
        currency_pos = message.find("*幣別曝險：*")
        trend_pos = message.find("*風險趨勢（7日）：*")

        assert risk_views_pos < currency_pos < trend_pos

    def test_existing_sections_unchanged(self):
        """Test that existing sections remain unchanged with risk trend."""
        params = self._create_base_message_params()
        params["concentration_trend"] = {
            "baseline_top1": Decimal("40"),
            "today_top1": Decimal("45"),
            "delta_top1": Decimal("5"),
            "baseline_top3": Decimal("75"),
            "today_top3": Decimal("80"),
            "delta_top3": Decimal("5"),
        }

        message = format_slack_message(**params)

        # Verify existing sections are present (zh-TW labels)
        assert "*每日損益報表 - 2026-01-20*" in message
        assert "總計：+100,000 TWD" in message
        assert "*今日損益：+5,000 TWD*" in message
        assert "*風險視圖：*" in message
        assert "*集中度風險（依市值）：*" in message
        assert "*幣別曝險：*" in message


class TestComputeCurrencyExposureFromPrices:
    """Tests for compute_currency_exposure_from_prices pure function."""

    def test_basic_currency_exposure(self):
        """Test basic currency exposure computation."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),  # 100 * 150 * 32 = 480000 TWD
            "TSLA": (Decimal("50"), "USD"),   # 50 * 200 * 32 = 320000 TWD
            "2330": (Decimal("1000"), "TWD"),  # 1000 * 600 = 600000 TWD
        }
        prices = {
            "AAPL": Decimal("150"),
            "TSLA": Decimal("200"),
            "2330": Decimal("600"),
        }
        usd_twd = Decimal("32")

        exposure = compute_currency_exposure_from_prices(
            symbol_positions, prices, usd_twd
        )

        # Total: 480000 + 320000 + 600000 = 1400000
        # USD: 800000 / 1400000 = 57.14%
        # TWD: 600000 / 1400000 = 42.86%
        assert len(exposure) == 2
        # Sorted by pct desc
        usd_item = next(e for e in exposure if e[0] == "USD")
        twd_item = next(e for e in exposure if e[0] == "TWD")
        assert abs(usd_item[1] - Decimal("57.14")) < Decimal("0.01")
        assert abs(twd_item[1] - Decimal("42.86")) < Decimal("0.01")

    def test_single_currency(self):
        """Test with single currency portfolio."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "TSLA": (Decimal("50"), "USD"),
        }
        prices = {
            "AAPL": Decimal("150"),
            "TSLA": Decimal("200"),
        }
        usd_twd = Decimal("32")

        exposure = compute_currency_exposure_from_prices(
            symbol_positions, prices, usd_twd
        )

        assert len(exposure) == 1
        assert exposure[0][0] == "USD"
        assert exposure[0][1] == Decimal("100")

    def test_skips_missing_prices(self):
        """Test that symbols without prices are skipped."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "UNKNOWN": (Decimal("50"), "TWD"),
        }
        prices = {"AAPL": Decimal("150")}
        usd_twd = Decimal("32")

        exposure = compute_currency_exposure_from_prices(
            symbol_positions, prices, usd_twd
        )

        assert len(exposure) == 1
        assert exposure[0][0] == "USD"


class TestComputeCurrencyExposureTrend:
    """Tests for compute_currency_exposure_trend function."""

    def test_trend_computation(self):
        """Test currency exposure trend with synthetic data."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "2330": (Decimal("100"), "TWD"),
            "0050": (Decimal("100"), "TWD"),
        }
        # Today: AAPL=150 (4800 TWD), 2330=500 (500 TWD), 0050=100 (100 TWD)
        # Total: 5400 TWD, USD: 88.89%, TWD: 11.11%
        today_prices = {
            "AAPL": Decimal("150"),
            "2330": Decimal("500"),
            "0050": Decimal("100"),
        }
        # Baseline: AAPL=100 (3200 TWD), 2330=600 (600 TWD), 0050=200 (200 TWD)
        # Total: 4000 TWD, USD: 80%, TWD: 20%
        baseline_prices = {
            "AAPL": Decimal("100"),
            "2330": Decimal("600"),
            "0050": Decimal("200"),
        }
        usd_twd = Decimal("32")

        trend = compute_currency_exposure_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd
        )

        assert trend is not None
        assert len(trend) == 2

        # Find USD and TWD entries
        usd_trend = next(t for t in trend if t["ccy"] == "USD")
        twd_trend = next(t for t in trend if t["ccy"] == "TWD")

        assert usd_trend["baseline_pct"] == Decimal("80")
        assert twd_trend["baseline_pct"] == Decimal("20")

        # Today calculations
        # AAPL: 100 * 150 * 32 = 480000, 2330: 100 * 500 = 50000, 0050: 100 * 100 = 10000
        # Total: 540000, USD: 480000/540000 = 88.89%, TWD: 60000/540000 = 11.11%
        assert abs(usd_trend["today_pct"] - Decimal("88.89")) < Decimal("0.01")
        assert abs(twd_trend["today_pct"] - Decimal("11.11")) < Decimal("0.01")

    def test_trend_with_currency_appearing_only_baseline(self):
        """Test trend when a currency exists only in baseline (e.g., sold all TWD)."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "2330": (Decimal("100"), "TWD"),
        }
        today_prices = {
            "AAPL": Decimal("150"),
            # 2330 not in today prices (position sold or delisted)
        }
        baseline_prices = {
            "AAPL": Decimal("100"),
            "2330": Decimal("500"),
        }
        usd_twd = Decimal("32")

        trend = compute_currency_exposure_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd,
            min_symbols_required=1
        )

        assert trend is not None
        # Today: only USD (100%)
        # Baseline: USD 86.49%, TWD 13.51% (approx)
        # Both currencies should be in result (union)
        currencies = {t["ccy"] for t in trend}
        assert "USD" in currencies
        assert "TWD" in currencies

    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None gracefully."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "2330": (Decimal("100"), "TWD"),
        }
        today_prices = {"AAPL": Decimal("150"), "2330": Decimal("500")}
        baseline_prices = {"AAPL": Decimal("100")}  # Only 1 symbol
        usd_twd = Decimal("32")

        trend = compute_currency_exposure_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd
        )

        assert trend is None

    def test_sorted_by_today_pct_descending(self):
        """Test that result is sorted by today's pct descending."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "2330": (Decimal("1000"), "TWD"),
            "0050": (Decimal("100"), "TWD"),
        }
        # TWD should be larger today
        today_prices = {
            "AAPL": Decimal("100"),  # 3200 TWD
            "2330": Decimal("600"),  # 600000 TWD
            "0050": Decimal("100"),  # 10000 TWD
        }
        baseline_prices = {
            "AAPL": Decimal("100"),
            "2330": Decimal("600"),
            "0050": Decimal("100"),
        }
        usd_twd = Decimal("32")

        trend = compute_currency_exposure_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd
        )

        assert trend is not None
        # TWD should be first (larger exposure)
        assert trend[0]["ccy"] == "TWD"
        assert trend[1]["ccy"] == "USD"


class TestFormatSlackMessageCurrencyExposureTrend:
    """Tests for Currency Exposure Trend section in format_slack_message."""

    def _create_base_message_params(self):
        """Create base parameters for format_slack_message."""
        return {
            "today": "2026-01-20",
            "usd_twd": Decimal("32.5"),
            "total_pnl": Decimal("100000"),
            "user_pnl": {"user1": Decimal("100000")},
            "top_symbols": [("AAPL", Decimal("50000"))],
            "symbols_count": 5,
            "missing_symbols": [],
            "lookup_attempts": {},
            "total_day_pnl": Decimal("5000"),
            "top_day_symbols": [("AAPL", Decimal("3000"))],
            "missing_prev_close": [],
            "concentration": [("AAPL", Decimal("50"), Decimal("500000"))],
            "currency_exposure": [("USD", Decimal("100"), Decimal("500000"))],
            "concentration_trend": {
                "baseline_top1": Decimal("40"),
                "today_top1": Decimal("45"),
                "delta_top1": Decimal("5"),
                "baseline_top3": Decimal("75"),
                "today_top3": Decimal("80"),
                "delta_top3": Decimal("5"),
            },
        }

    def test_currency_exposure_trend_present(self):
        """Test that currency exposure trend section is present when data available."""
        params = self._create_base_message_params()
        params["currency_exposure_trend"] = [
            {
                "ccy": "USD",
                "baseline_pct": Decimal("60.5"),
                "today_pct": Decimal("55.2"),
                "delta_pct": Decimal("-5.3"),
            },
            {
                "ccy": "TWD",
                "baseline_pct": Decimal("39.5"),
                "today_pct": Decimal("44.8"),
                "delta_pct": Decimal("5.3"),
            },
        ]

        message = format_slack_message(**params)

        assert "*幣別曝險趨勢（7日）：*" in message
        assert "USD" in message
        assert "60.5% → 55.2%" in message
        assert "-5.3%" in message
        assert "TWD" in message
        assert "39.5% → 44.8%" in message
        assert "+5.3%" in message

    def test_currency_exposure_trend_unavailable(self):
        """Test fallback message when currency exposure trend unavailable."""
        params = self._create_base_message_params()
        params["currency_exposure_trend"] = None

        message = format_slack_message(**params)

        assert "*幣別曝險趨勢（7日）：*" in message
        assert "無法取得足夠的歷史價格，略過。" in message

    def test_currency_exposure_trend_appended_after_concentration_trend(self):
        """Test that currency exposure trend appears after concentration trend."""
        params = self._create_base_message_params()
        params["currency_exposure_trend"] = [
            {
                "ccy": "USD",
                "baseline_pct": Decimal("60"),
                "today_pct": Decimal("55"),
                "delta_pct": Decimal("-5"),
            },
        ]

        message = format_slack_message(**params)

        concentration_trend_pos = message.find("*風險趨勢（7日）：*")
        currency_trend_pos = message.find("*幣別曝險趨勢（7日）：*")

        assert concentration_trend_pos < currency_trend_pos


class TestGetTopNSymbolsWithCcy:
    """Tests for D-lite get_top_n_symbols_with_ccy function."""

    def test_returns_top_n_symbols(self):
        """Test that function returns top N symbols by market value."""
        concentration = [
            ("A", Decimal("40"), Decimal("400000")),
            ("B", Decimal("30"), Decimal("300000")),
            ("C", Decimal("20"), Decimal("200000")),
            ("D", Decimal("10"), Decimal("100000")),
        ]
        symbol_positions = {
            "A": (Decimal("100"), "USD"),
            "B": (Decimal("200"), "TWD"),
            "C": (Decimal("300"), "USD"),
            "D": (Decimal("400"), "TWD"),
        }

        result = get_top_n_symbols_with_ccy(concentration, symbol_positions, n=3)

        assert len(result) == 3
        assert result[0] == ("A", "USD")
        assert result[1] == ("B", "TWD")
        assert result[2] == ("C", "USD")

    def test_returns_all_when_n_exceeds_count(self):
        """Test that function returns all symbols when N exceeds total count."""
        concentration = [
            ("A", Decimal("60"), Decimal("600000")),
            ("B", Decimal("40"), Decimal("400000")),
        ]
        symbol_positions = {
            "A": (Decimal("100"), "USD"),
            "B": (Decimal("200"), "TWD"),
        }

        result = get_top_n_symbols_with_ccy(concentration, symbol_positions, n=10)

        assert len(result) == 2
        assert result[0] == ("A", "USD")
        assert result[1] == ("B", "TWD")

    def test_skips_symbols_not_in_positions(self):
        """Test that symbols not in symbol_positions are skipped."""
        concentration = [
            ("A", Decimal("50"), Decimal("500000")),
            ("MISSING", Decimal("30"), Decimal("300000")),
            ("B", Decimal("20"), Decimal("200000")),
        ]
        symbol_positions = {
            "A": (Decimal("100"), "USD"),
            "B": (Decimal("200"), "TWD"),
            # MISSING not in positions
        }

        result = get_top_n_symbols_with_ccy(concentration, symbol_positions, n=3)

        assert len(result) == 2
        assert ("MISSING", "USD") not in result
        assert ("MISSING", "TWD") not in result

    def test_default_n_is_top_n_for_baseline(self):
        """Test that default N uses TOP_N_FOR_BASELINE constant."""
        # Create concentration list with more symbols than TOP_N_FOR_BASELINE
        concentration = [
            (f"SYM{i}", Decimal(str(100 - i)), Decimal(str((100 - i) * 10000)))
            for i in range(15)
        ]
        symbol_positions = {
            f"SYM{i}": (Decimal("100"), "TWD")
            for i in range(15)
        }

        result = get_top_n_symbols_with_ccy(concentration, symbol_positions)

        assert len(result) == TOP_N_FOR_BASELINE


class TestDLiteBaselineFetch:
    """Tests for D-lite baseline fetch behavior with mocked yfinance."""

    @patch("services.api.daily_close.yf.Ticker")
    def test_baseline_computed_with_multiple_rows(self, mock_ticker_class):
        """Test baseline is computed when history returns multiple rows."""
        import pandas as pd

        # Setup mock with multiple rows (typical case)
        mock_ticker = MagicMock()
        mock_hist = pd.DataFrame({
            "Close": [145.0, 148.0, 150.0],
        }, index=pd.to_datetime(["2026-01-10", "2026-01-11", "2026-01-12"]))
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker

        symbols_with_ccy = [("AAPL", "USD")]
        target_date = date(2026, 1, 13)

        prices, missing = fetch_historical_prices(symbols_with_ccy, target_date)

        # Should get the last available close (150.0)
        assert "AAPL" in prices
        assert prices["AAPL"] == Decimal("150")
        assert missing == []

    @patch("services.api.daily_close.yf.Ticker")
    def test_baseline_computed_with_single_row(self, mock_ticker_class):
        """Test baseline uses single row if that's all available."""
        import pandas as pd

        # Setup mock with only 1 row (edge case)
        mock_ticker = MagicMock()
        mock_hist = pd.DataFrame({
            "Close": [150.0],
        }, index=pd.to_datetime(["2026-01-12"]))
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker

        symbols_with_ccy = [("AAPL", "USD")]
        target_date = date(2026, 1, 13)

        prices, missing = fetch_historical_prices(symbols_with_ccy, target_date)

        # Single row should still be used as baseline
        assert "AAPL" in prices
        assert prices["AAPL"] == Decimal("150")
        assert missing == []

    @patch("services.api.daily_close.yf.Ticker")
    def test_baseline_missing_when_empty_history(self, mock_ticker_class):
        """Test baseline is missing when history returns empty DataFrame."""
        import pandas as pd

        # Setup mock with empty history
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        symbols_with_ccy = [("NEWIPO", "USD")]
        target_date = date(2026, 1, 13)

        prices, missing = fetch_historical_prices(symbols_with_ccy, target_date)

        # Symbol should be in missing list
        assert prices == {}
        assert "NEWIPO" in missing

    @patch("services.api.daily_close.yf.Ticker")
    def test_baseline_missing_when_all_nan(self, mock_ticker_class):
        """Test baseline is missing when all Close values are NaN."""
        import pandas as pd
        import numpy as np

        mock_ticker = MagicMock()
        mock_hist = pd.DataFrame({
            "Close": [np.nan, np.nan, np.nan],
        }, index=pd.to_datetime(["2026-01-10", "2026-01-11", "2026-01-12"]))
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker

        symbols_with_ccy = [("BADDATA", "USD")]
        target_date = date(2026, 1, 13)

        prices, missing = fetch_historical_prices(symbols_with_ccy, target_date)

        # All NaN means no valid close, should be missing
        assert prices == {}
        assert "BADDATA" in missing

    def test_concentration_trend_with_top_n_subset(self):
        """Test concentration trend computation with Top-N subset."""
        # Simulate D-lite: only top 3 symbols have baseline prices
        symbol_positions = {
            "A": (Decimal("100"), "TWD"),
            "B": (Decimal("100"), "TWD"),
            "C": (Decimal("100"), "TWD"),
            "D": (Decimal("100"), "TWD"),  # Not in baseline (not top N)
        }
        today_prices = {
            "A": Decimal("50"),
            "B": Decimal("30"),
            "C": Decimal("15"),
            "D": Decimal("5"),
        }
        # Only top 3 have baseline (simulating D-lite fetch)
        baseline_prices = {
            "A": Decimal("40"),
            "B": Decimal("35"),
            "C": Decimal("20"),
            # D missing - not in top N
        }
        usd_twd = Decimal("32")

        trend = compute_concentration_trend(
            symbol_positions, today_prices, baseline_prices, usd_twd
        )

        # Should succeed with 3 symbols
        assert trend is not None
        # Trend computed from symbols with both prices (A, B, C)
        assert trend["today_top1"] is not None
        assert trend["baseline_top1"] is not None
