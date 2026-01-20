# packages/pnl/tests/test_risk_views.py
"""Unit tests for Sprint C Risk Views (Concentration Risk & Currency Exposure)."""
from decimal import Decimal

from services.api.daily_close import compute_risk_views, format_slack_message


class TestComputeRiskViews:
    """Tests for compute_risk_views function."""

    def test_concentration_pct_sums_to_100(self):
        """
        Test that concentration percentages sum to approximately 100%.

        Uses a fixture portfolio with multiple symbols.
        """
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "2330": (Decimal("1000"), "TWD"),
            "TSLA": (Decimal("50"), "USD"),
            "0050": (Decimal("500"), "TWD"),
        }
        prices = {
            "AAPL": Decimal("150"),   # 150 * 100 = 15000 USD -> 480000 TWD
            "2330": Decimal("600"),   # 600 * 1000 = 600000 TWD
            "TSLA": Decimal("200"),   # 200 * 50 = 10000 USD -> 320000 TWD
            "0050": Decimal("120"),   # 120 * 500 = 60000 TWD
        }
        usd_twd = Decimal("32")

        total_portfolio, concentration, currency_exposure = compute_risk_views(
            symbol_positions, prices, usd_twd
        )

        # Sum of all concentration percentages should be ~100%
        total_pct = sum(pct for _, pct, _ in concentration)
        assert abs(total_pct - Decimal("100")) < Decimal("0.01"), f"Expected ~100%, got {total_pct}"

    def test_concentration_sorted_descending(self):
        """Test that concentration list is sorted by percentage descending."""
        symbol_positions = {
            "A": (Decimal("10"), "TWD"),
            "B": (Decimal("30"), "TWD"),
            "C": (Decimal("20"), "TWD"),
        }
        prices = {
            "A": Decimal("100"),  # 1000 TWD
            "B": Decimal("100"),  # 3000 TWD
            "C": Decimal("100"),  # 2000 TWD
        }
        usd_twd = Decimal("32")

        _, concentration, _ = compute_risk_views(symbol_positions, prices, usd_twd)

        # B (50%), C (33.33%), A (16.67%)
        assert concentration[0][0] == "B"
        assert concentration[1][0] == "C"
        assert concentration[2][0] == "A"

        # Verify sorted descending
        for i in range(len(concentration) - 1):
            assert concentration[i][1] >= concentration[i + 1][1]

    def test_currency_exposure_aggregation(self):
        """Test that currency exposure correctly aggregates symbols by currency."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "TSLA": (Decimal("50"), "USD"),
            "2330": (Decimal("1000"), "TWD"),
            "0050": (Decimal("500"), "TWD"),
        }
        prices = {
            "AAPL": Decimal("150"),   # 15000 USD -> 480000 TWD
            "TSLA": Decimal("200"),   # 10000 USD -> 320000 TWD
            "2330": Decimal("600"),   # 600000 TWD
            "0050": Decimal("120"),   # 60000 TWD
        }
        usd_twd = Decimal("32")

        total_portfolio, _, currency_exposure = compute_risk_views(
            symbol_positions, prices, usd_twd
        )

        # Expected USD: (15000 + 10000) * 32 = 800000 TWD
        # Expected TWD: 600000 + 60000 = 660000 TWD
        # Total: 800000 + 660000 = 1460000 TWD

        assert total_portfolio == Decimal("1460000")

        # Convert to dict for easier assertions
        exposure_dict = {ccy: (pct, val) for ccy, pct, val in currency_exposure}

        assert "USD" in exposure_dict
        assert "TWD" in exposure_dict
        assert exposure_dict["USD"][1] == Decimal("800000")
        assert exposure_dict["TWD"][1] == Decimal("660000")

    def test_currency_exposure_sorted_descending(self):
        """Test that currency exposure is sorted by percentage descending."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "2330": (Decimal("1000"), "TWD"),
        }
        prices = {
            "AAPL": Decimal("150"),   # 15000 USD -> 480000 TWD
            "2330": Decimal("600"),   # 600000 TWD
        }
        usd_twd = Decimal("32")

        _, _, currency_exposure = compute_risk_views(
            symbol_positions, prices, usd_twd
        )

        # TWD (600000) should be first, USD (480000) second
        assert currency_exposure[0][0] == "TWD"
        assert currency_exposure[1][0] == "USD"

        # Verify sorted descending
        for i in range(len(currency_exposure) - 1):
            assert currency_exposure[i][1] >= currency_exposure[i + 1][1]

    def test_currency_exposure_pct_sums_to_100(self):
        """Test that currency exposure percentages sum to ~100%."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "TSLA": (Decimal("50"), "USD"),
            "2330": (Decimal("1000"), "TWD"),
        }
        prices = {
            "AAPL": Decimal("150"),
            "TSLA": Decimal("200"),
            "2330": Decimal("600"),
        }
        usd_twd = Decimal("32")

        _, _, currency_exposure = compute_risk_views(
            symbol_positions, prices, usd_twd
        )

        total_pct = sum(pct for _, pct, _ in currency_exposure)
        assert abs(total_pct - Decimal("100")) < Decimal("0.01"), f"Expected ~100%, got {total_pct}"

    def test_skips_symbols_without_price(self):
        """Test that symbols without price data are skipped."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "UNKNOWN": (Decimal("50"), "USD"),  # No price data
        }
        prices = {
            "AAPL": Decimal("150"),
            # UNKNOWN missing
        }
        usd_twd = Decimal("32")

        total_portfolio, concentration, currency_exposure = compute_risk_views(
            symbol_positions, prices, usd_twd
        )

        # Only AAPL should be counted: 15000 * 32 = 480000 TWD
        assert total_portfolio == Decimal("480000")
        assert len(concentration) == 1
        assert concentration[0][0] == "AAPL"
        assert concentration[0][1] == Decimal("100")  # 100% concentration

    def test_skips_zero_quantity(self):
        """Test that symbols with zero quantity are skipped."""
        symbol_positions = {
            "AAPL": (Decimal("100"), "USD"),
            "SOLD": (Decimal("0"), "USD"),  # Zero quantity
        }
        prices = {
            "AAPL": Decimal("150"),
            "SOLD": Decimal("100"),
        }
        usd_twd = Decimal("32")

        _, concentration, _ = compute_risk_views(
            symbol_positions, prices, usd_twd
        )

        assert len(concentration) == 1
        assert concentration[0][0] == "AAPL"

    def test_empty_portfolio(self):
        """Test handling of empty portfolio."""
        symbol_positions = {}
        prices = {}
        usd_twd = Decimal("32")

        total_portfolio, concentration, currency_exposure = compute_risk_views(
            symbol_positions, prices, usd_twd
        )

        assert total_portfolio == Decimal("0")
        assert concentration == []
        assert currency_exposure == []


class TestFormatSlackMessageRiskViews:
    """Tests for Risk Views section in format_slack_message."""

    def _create_base_message_params(self):
        """Create base parameters for format_slack_message."""
        return {
            "today": "2026-01-19",
            "usd_twd": Decimal("32.5"),
            "total_pnl": Decimal("100000"),
            "user_pnl": {"user1": Decimal("100000")},
            "top_symbols": [("AAPL", Decimal("50000")), ("TSLA", Decimal("30000"))],
            "symbols_count": 10,
            "missing_symbols": [],
            "lookup_attempts": {},
            "total_day_pnl": Decimal("5000"),
            "top_day_symbols": [("AAPL", Decimal("3000"))],
            "missing_prev_close": [],
        }

    def test_report_contains_risk_views_section(self):
        """Test that report contains Risk Views section when data is provided."""
        params = self._create_base_message_params()
        params["concentration"] = [
            ("AAPL", Decimal("40.5"), Decimal("405000")),
            ("TSLA", Decimal("30.3"), Decimal("303000")),
            ("2330", Decimal("29.2"), Decimal("292000")),
        ]
        params["currency_exposure"] = [
            ("USD", Decimal("70.8"), Decimal("708000")),
            ("TWD", Decimal("29.2"), Decimal("292000")),
        ]

        message = format_slack_message(**params)

        assert "*風險視圖：*" in message
        assert "*集中度風險（依市值）：*" in message
        assert "*幣別曝險：*" in message

    def test_concentration_risk_formatting(self):
        """Test concentration risk formatting with correct percentage and value."""
        params = self._create_base_message_params()
        params["concentration"] = [
            ("AAPL", Decimal("40.5"), Decimal("405000")),
            ("TSLA", Decimal("30.3"), Decimal("303000")),
        ]
        params["currency_exposure"] = [
            ("USD", Decimal("100"), Decimal("708000")),
        ]

        message = format_slack_message(**params)

        # Check formatting: pct% with 1 decimal, value_twd with thousands separators (zh-TW)
        assert "1. AAPL：40.5%（405,000）" in message
        assert "2. TSLA：30.3%（303,000）" in message

    def test_currency_exposure_formatting(self):
        """Test currency exposure formatting."""
        params = self._create_base_message_params()
        params["concentration"] = [
            ("AAPL", Decimal("100"), Decimal("1000000")),
        ]
        params["currency_exposure"] = [
            ("USD", Decimal("70.8"), Decimal("708000")),
            ("TWD", Decimal("29.2"), Decimal("292000")),
        ]

        message = format_slack_message(**params)

        assert "- USD：70.8%（708,000）" in message
        assert "- TWD：29.2%（292,000）" in message

    def test_top1_concentration_risk_alert(self):
        """Test Top-1 concentration risk alert when top1 > 25%."""
        params = self._create_base_message_params()
        params["concentration"] = [
            ("AAPL", Decimal("30.0"), Decimal("300000")),  # > 25%
            ("TSLA", Decimal("25.0"), Decimal("250000")),
            ("2330", Decimal("45.0"), Decimal("450000")),
        ]
        params["currency_exposure"] = [
            ("USD", Decimal("100"), Decimal("1000000")),
        ]

        message = format_slack_message(**params)

        assert "單一標的集中度風險" in message

    def test_no_top1_alert_when_under_threshold(self):
        """Test no Top-1 alert when top1 <= 25%."""
        params = self._create_base_message_params()
        params["concentration"] = [
            ("AAPL", Decimal("25.0"), Decimal("250000")),  # == 25%, not > 25%
            ("TSLA", Decimal("25.0"), Decimal("250000")),
            ("2330", Decimal("25.0"), Decimal("250000")),
            ("0050", Decimal("25.0"), Decimal("250000")),
        ]
        params["currency_exposure"] = [
            ("TWD", Decimal("100"), Decimal("1000000")),
        ]

        message = format_slack_message(**params)

        assert "單一標的集中度風險" not in message

    def test_top3_concentration_risk_alert(self):
        """Test Top-3 concentration risk alert when sum(top3) > 60%."""
        params = self._create_base_message_params()
        params["concentration"] = [
            ("AAPL", Decimal("22.0"), Decimal("220000")),
            ("TSLA", Decimal("21.0"), Decimal("210000")),
            ("2330", Decimal("20.0"), Decimal("200000")),  # sum = 63% > 60%
            ("0050", Decimal("37.0"), Decimal("370000")),
        ]
        params["currency_exposure"] = [
            ("USD", Decimal("100"), Decimal("1000000")),
        ]

        message = format_slack_message(**params)

        assert "前三大集中度風險" in message

    def test_no_top3_alert_when_under_threshold(self):
        """Test no Top-3 alert when sum(top3) <= 60%."""
        params = self._create_base_message_params()
        params["concentration"] = [
            ("AAPL", Decimal("20.0"), Decimal("200000")),
            ("TSLA", Decimal("20.0"), Decimal("200000")),
            ("2330", Decimal("20.0"), Decimal("200000")),  # sum = 60%, not > 60%
            ("0050", Decimal("40.0"), Decimal("400000")),
        ]
        params["currency_exposure"] = [
            ("TWD", Decimal("100"), Decimal("1000000")),
        ]

        message = format_slack_message(**params)

        assert "前三大集中度風險" not in message

    def test_existing_pnl_sections_unchanged(self):
        """Test that existing P&L sections remain unchanged with risk views."""
        params = self._create_base_message_params()
        params["concentration"] = [
            ("AAPL", Decimal("100"), Decimal("1000000")),
        ]
        params["currency_exposure"] = [
            ("USD", Decimal("100"), Decimal("1000000")),
        ]

        message = format_slack_message(**params)

        # Verify existing sections are present (zh-TW labels)
        assert "*每日損益報表 - 2026-01-19*" in message
        assert "USD/TWD：32.50" in message
        assert "計入標的數：10" in message
        assert "*用戶損益排名（前3名）：*" in message
        assert "*前5大標的（累計）：*" in message
        assert "總計：+100,000 TWD" in message
        assert "*今日損益：+5,000 TWD*" in message
        assert "*前5大標的（今日）：*" in message
        assert "缺少報價（累計）：無" in message

    def test_no_risk_views_when_params_none(self):
        """Test that Risk Views section is not added when params are None."""
        params = self._create_base_message_params()
        # concentration and currency_exposure default to None

        message = format_slack_message(**params)

        assert "*風險視圖：*" not in message
        assert "*集中度風險" not in message
        assert "*幣別曝險：*" not in message

    def test_top5_symbols_shown(self):
        """Test that only top 5 symbols are shown in concentration risk."""
        params = self._create_base_message_params()
        params["concentration"] = [
            ("A", Decimal("20"), Decimal("200000")),
            ("B", Decimal("18"), Decimal("180000")),
            ("C", Decimal("16"), Decimal("160000")),
            ("D", Decimal("14"), Decimal("140000")),
            ("E", Decimal("12"), Decimal("120000")),
            ("F", Decimal("10"), Decimal("100000")),  # 6th, should not appear
            ("G", Decimal("10"), Decimal("100000")),  # 7th, should not appear
        ]
        params["currency_exposure"] = [
            ("USD", Decimal("100"), Decimal("1000000")),
        ]

        message = format_slack_message(**params)

        assert "1. A：" in message
        assert "2. B：" in message
        assert "3. C：" in message
        assert "4. D：" in message
        assert "5. E：" in message
        # F and G should not appear in the numbered list
        assert "6. F：" not in message
        assert "7. G：" not in message


class TestRiskViewsIntegration:
    """Integration tests for risk views end-to-end."""

    def test_full_report_structure_with_risk_views(self):
        """Test full report structure with all sections including risk views."""
        message = format_slack_message(
            today="2026-01-19",
            usd_twd=Decimal("32"),
            total_pnl=Decimal("50000"),
            user_pnl={"alice": Decimal("30000"), "bob": Decimal("20000")},
            top_symbols=[("AAPL", Decimal("30000")), ("TSLA", Decimal("20000"))],
            symbols_count=5,
            missing_symbols=["UNKNOWN"],
            lookup_attempts={"UNKNOWN": [("UNKNOWN", "fail")]},
            total_day_pnl=Decimal("2000"),
            top_day_symbols=[("AAPL", Decimal("1500")), ("TSLA", Decimal("500"))],
            missing_prev_close=[],
            concentration=[
                ("AAPL", Decimal("60"), Decimal("600000")),
                ("TSLA", Decimal("40"), Decimal("400000")),
            ],
            currency_exposure=[
                ("USD", Decimal("100"), Decimal("1000000")),
            ],
        )

        # Verify section order: existing sections come before risk views (zh-TW labels)
        pnl_section_pos = message.find("總計：")
        day_pnl_pos = message.find("*今日損益：")
        missing_pos = message.find("缺少報價（累計）：")
        risk_views_pos = message.find("*風險視圖：*")

        assert pnl_section_pos < day_pnl_pos
        assert day_pnl_pos < missing_pos
        assert missing_pos < risk_views_pos

        # Risk Views appears at the end (append-only)
        concentration_pos = message.find("*集中度風險")
        currency_pos = message.find("*幣別曝險：*")

        assert risk_views_pos < concentration_pos
        assert concentration_pos < currency_pos


class TestDisplaySymbol:
    """Tests for Sprint E display_symbol function."""

    def test_display_symbol_mapped_ticker(self):
        """Test that mapped ticker returns 'SYMBOL（NAME）' format."""
        from services.api.daily_close import display_symbol

        result = display_symbol("0050")
        assert result == "0050（元大台灣50）"

        result = display_symbol("2330")
        assert result == "2330（台積電）"

        result = display_symbol("00983A")
        assert result == "00983A（中信美10Y+A公司債）"

    def test_display_symbol_unmapped_ticker(self):
        """Test that unmapped ticker returns original symbol unchanged."""
        from services.api.daily_close import display_symbol

        result = display_symbol("AAPL")
        assert result == "AAPL"

        result = display_symbol("TSLA")
        assert result == "TSLA"

        result = display_symbol("UNKNOWN")
        assert result == "UNKNOWN"

    def test_display_symbol_does_not_affect_underlying(self):
        """Test that display_symbol is display-only and does not modify the symbol key."""
        from services.api.daily_close import display_symbol, SYMBOL_NAMES_ZH

        symbol = "0050"
        display = display_symbol(symbol)

        # Symbol key is unchanged
        assert symbol == "0050"

        # Display shows name
        assert display == "0050（元大台灣50）"

        # Underlying dict unchanged
        assert "0050" in SYMBOL_NAMES_ZH
        assert SYMBOL_NAMES_ZH["0050"] == "元大台灣50"
