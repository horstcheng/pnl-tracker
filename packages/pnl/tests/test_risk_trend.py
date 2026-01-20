# packages/pnl/tests/test_risk_trend.py
"""Unit tests for Sprint D Concentration Risk Trend (7D, no snapshot)."""
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from services.api.daily_close import (
    compute_concentration_weights,
    compute_concentration_from_prices,
    compute_concentration_trend,
    fetch_historical_prices,
    format_slack_message,
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
