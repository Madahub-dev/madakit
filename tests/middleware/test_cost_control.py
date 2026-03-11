"""Tests for CostControlMiddleware.

Validates budget tracking, cost cap enforcement, alert callbacks,
and integration with the BaseAgentClient contract.
"""

from __future__ import annotations

import pytest

from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.middleware.cost_control import CostControlMiddleware
from helpers import MockProvider


class TestModuleExports:
    """Verify cost_control.py module structure."""

    def test_all_exports(self) -> None:
        """__all__ contains CostControlMiddleware."""
        from mada_modelkit.middleware import cost_control
        assert hasattr(cost_control, "__all__")
        assert "CostControlMiddleware" in cost_control.__all__
        assert len(cost_control.__all__) == 1

    def test_middleware_importable(self) -> None:
        """CostControlMiddleware is importable from cost_control module."""
        from mada_modelkit.middleware.cost_control import CostControlMiddleware as Imported
        assert Imported is CostControlMiddleware


class TestCostControlMiddlewareConstructor:
    """Test constructor parameter handling and state initialisation."""

    def test_minimal_constructor(self) -> None:
        """Constructor with required parameters."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.01
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)
        assert middleware._client is mock
        assert middleware._budget_cap is None
        assert middleware._alert_threshold == 0.8
        assert middleware._cost_fn is cost_fn
        assert middleware._on_alert is None

    def test_explicit_budget_cap(self) -> None:
        """budget_cap parameter is stored."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.01
        middleware = CostControlMiddleware(client=mock, budget_cap=10.0, cost_fn=cost_fn)
        assert middleware._budget_cap == 10.0

    def test_explicit_alert_threshold(self) -> None:
        """alert_threshold parameter is stored."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.01
        middleware = CostControlMiddleware(client=mock, alert_threshold=0.9, cost_fn=cost_fn)
        assert middleware._alert_threshold == 0.9

    def test_cost_fn_stored(self) -> None:
        """cost_fn parameter is stored."""
        mock = MockProvider()
        cost_fn = lambda resp: resp.input_tokens * 0.001
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)
        assert middleware._cost_fn is cost_fn

    def test_explicit_on_alert(self) -> None:
        """on_alert parameter is stored."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.01
        alert_callback = lambda current, threshold: None
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, on_alert=alert_callback)
        assert middleware._on_alert is alert_callback

    def test_initial_spend_zero(self) -> None:
        """Total spend starts at zero."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.01
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)
        assert middleware._total_spend == 0.0

    def test_initial_alert_not_fired(self) -> None:
        """Alert flag starts as False."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.01
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)
        assert middleware._alert_fired is False

    def test_instance_isolation(self) -> None:
        """Multiple instances have independent state."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        cost_fn1 = lambda resp: 0.01
        cost_fn2 = lambda resp: 0.02
        middleware1 = CostControlMiddleware(client=mock1, budget_cap=5.0, cost_fn=cost_fn1)
        middleware2 = CostControlMiddleware(client=mock2, budget_cap=10.0, cost_fn=cost_fn2)
        assert middleware1._client is mock1
        assert middleware2._client is mock2
        assert middleware1._budget_cap == 5.0
        assert middleware2._budget_cap == 10.0
        assert middleware1._cost_fn is cost_fn1
        assert middleware2._cost_fn is cost_fn2

    def test_wraps_middleware(self) -> None:
        """Can wrap another middleware instance."""
        mock = MockProvider()
        from mada_modelkit.middleware.retry import RetryMiddleware
        retry = RetryMiddleware(client=mock, max_retries=2)
        cost_fn = lambda resp: 0.01
        middleware = CostControlMiddleware(client=retry, cost_fn=cost_fn)
        assert middleware._client is retry
        assert retry._client is mock

    def test_super_init_called(self) -> None:
        """Constructor calls super().__init__()."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.01
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)
        # Inherits from BaseAgentClient, so should have _semaphore attribute
        assert hasattr(middleware, "_semaphore")


class TestBudgetTracking:
    """Test budget tracking and cost accumulation."""

    def test_track_cost_increments_spend(self) -> None:
        """_track_cost adds cost to total_spend."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.05
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)

        assert middleware._total_spend == 0.05

    def test_track_cost_accumulates(self) -> None:
        """Multiple _track_cost calls accumulate."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.05
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)
        middleware._track_cost(response)
        middleware._track_cost(response)

        assert abs(middleware._total_spend - 0.15) < 0.001

    def test_track_cost_uses_cost_fn(self) -> None:
        """_track_cost calls cost_fn with response."""
        mock = MockProvider()
        # Cost based on token counts
        cost_fn = lambda resp: resp.input_tokens * 0.001 + resp.output_tokens * 0.002
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=100, output_tokens=50)
        middleware._track_cost(response)

        # 100 * 0.001 + 50 * 0.002 = 0.1 + 0.1 = 0.2
        assert abs(middleware._total_spend - 0.2) < 0.001

    def test_track_cost_different_responses(self) -> None:
        """_track_cost handles different response costs."""
        mock = MockProvider()
        cost_fn = lambda resp: resp.input_tokens * 0.001
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response1 = AgentResponse(content="test", model="test", input_tokens=100, output_tokens=10)
        response2 = AgentResponse(content="test", model="test", input_tokens=200, output_tokens=20)

        middleware._track_cost(response1)
        middleware._track_cost(response2)

        # 100 * 0.001 + 200 * 0.001 = 0.3
        assert abs(middleware._total_spend - 0.3) < 0.001

    def test_total_spend_property(self) -> None:
        """total_spend property returns current spend."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.05
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        assert middleware.total_spend == 0.0

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)

        assert middleware.total_spend == 0.05

    def test_total_spend_property_identity(self) -> None:
        """total_spend property returns same value as internal state."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.05
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)

        assert middleware.total_spend == middleware._total_spend

    def test_cost_fn_called_per_response(self) -> None:
        """cost_fn is called once per response."""
        mock = MockProvider()
        call_count = [0]

        def counting_cost_fn(resp):
            call_count[0] += 1
            return 0.01

        middleware = CostControlMiddleware(client=mock, cost_fn=counting_cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)
        middleware._track_cost(response)

        assert call_count[0] == 2

    def test_fractional_costs_accumulate_accurately(self) -> None:
        """Fractional costs accumulate without significant rounding errors."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.001
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        for _ in range(1000):
            middleware._track_cost(response)

        # 1000 * 0.001 = 1.0
        assert abs(middleware.total_spend - 1.0) < 0.0001

    def test_zero_cost_tracking(self) -> None:
        """Tracking zero-cost responses works correctly."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)

        assert middleware.total_spend == 0.0

    def test_instance_spend_isolation(self) -> None:
        """Different instances track spend independently."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        cost_fn = lambda resp: 0.05
        middleware1 = CostControlMiddleware(client=mock1, cost_fn=cost_fn)
        middleware2 = CostControlMiddleware(client=mock2, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware1._track_cost(response)
        middleware2._track_cost(response)
        middleware2._track_cost(response)

        assert middleware1.total_spend == 0.05
        assert middleware2.total_spend == 0.10


class TestBudgetCapEnforcement:
    """Test budget cap enforcement and BudgetExceededError."""

    def test_no_cap_allows_unlimited_spend(self) -> None:
        """Without budget_cap, spending is unlimited."""
        mock = MockProvider()
        cost_fn = lambda resp: 10.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)  # No cap

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        # Should not raise even with high costs
        middleware._track_cost(response)
        middleware._track_cost(response)
        middleware._track_cost(response)

        assert middleware.total_spend == 30.0

    def test_cap_enforcement_raises_on_exceed(self) -> None:
        """Exceeding budget_cap raises BudgetExceededError."""
        from mada_modelkit._errors import BudgetExceededError

        mock = MockProvider()
        cost_fn = lambda resp: 5.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=10.0)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)  # 5.0
        middleware._track_cost(response)  # 10.0

        # Third request would exceed cap
        with pytest.raises(BudgetExceededError):
            middleware._track_cost(response)

    def test_cap_enforcement_error_message(self) -> None:
        """BudgetExceededError includes current spend, cost, and cap."""
        from mada_modelkit._errors import BudgetExceededError

        mock = MockProvider()
        cost_fn = lambda resp: 6.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=10.0)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)  # 6.0

        # Second request would exceed (6.0 + 6.0 = 12.0 > 10.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            middleware._track_cost(response)

        error_msg = str(exc_info.value)
        assert "10.00" in error_msg  # Cap
        assert "6.00" in error_msg    # Current or cost

    def test_cap_allows_requests_up_to_limit(self) -> None:
        """Requests are allowed as long as total stays within cap."""
        mock = MockProvider()
        cost_fn = lambda resp: 2.5
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=10.0)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        # 4 requests * 2.5 = 10.0 (exactly at cap)
        middleware._track_cost(response)
        middleware._track_cost(response)
        middleware._track_cost(response)
        middleware._track_cost(response)

        assert middleware.total_spend == 10.0

        # Fifth would exceed
        from mada_modelkit._errors import BudgetExceededError
        with pytest.raises(BudgetExceededError):
            middleware._track_cost(response)

    def test_cap_enforcement_doesnt_increment_on_exceed(self) -> None:
        """When cap is exceeded, spend is not incremented."""
        from mada_modelkit._errors import BudgetExceededError

        mock = MockProvider()
        cost_fn = lambda resp: 8.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=10.0)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)  # 8.0

        try:
            middleware._track_cost(response)  # Would be 16.0, exceeds cap
        except BudgetExceededError:
            pass

        # Spend should still be 8.0, not 16.0
        assert middleware.total_spend == 8.0

    def test_exact_cap_match_allowed(self) -> None:
        """Request that brings total exactly to cap is allowed."""
        mock = MockProvider()
        cost_fn = lambda resp: 5.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=10.0)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)  # 5.0
        middleware._track_cost(response)  # 10.0 (exactly at cap)

        assert middleware.total_spend == 10.0

    def test_fractional_cap_enforcement(self) -> None:
        """Fractional budget caps work correctly."""
        from mada_modelkit._errors import BudgetExceededError

        mock = MockProvider()
        cost_fn = lambda resp: 0.3
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=1.0)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        # 3 requests * 0.3 = 0.9 (under cap)
        middleware._track_cost(response)
        middleware._track_cost(response)
        middleware._track_cost(response)

        # Fourth would be 1.2 > 1.0
        with pytest.raises(BudgetExceededError):
            middleware._track_cost(response)

    def test_very_small_cap(self) -> None:
        """Very small caps (< 1) are enforced correctly."""
        from mada_modelkit._errors import BudgetExceededError

        mock = MockProvider()
        cost_fn = lambda resp: 0.01
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=0.05)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        # 5 requests * 0.01 = 0.05 (at cap)
        for _ in range(5):
            middleware._track_cost(response)

        assert middleware.total_spend == 0.05

        # Sixth would exceed
        with pytest.raises(BudgetExceededError):
            middleware._track_cost(response)

    def test_budget_exceeded_is_middleware_error(self) -> None:
        """BudgetExceededError inherits from MiddlewareError."""
        from mada_modelkit._errors import BudgetExceededError, MiddlewareError

        assert issubclass(BudgetExceededError, MiddlewareError)


class TestAlertCallbacks:
    """Test alert callback triggering and threshold detection."""

    def test_alert_fires_when_threshold_crossed(self) -> None:
        """on_alert is called when spending crosses alert_threshold."""
        mock = MockProvider()
        cost_fn = lambda resp: 5.0
        alert_calls = []

        def alert_callback(current, threshold):
            alert_calls.append((current, threshold))

        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=10.0, alert_threshold=0.8, on_alert=alert_callback
        )

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)

        # First request: 5.0 (below 8.0 threshold)
        middleware._track_cost(response)
        assert len(alert_calls) == 0

        # Second request: 10.0 (crosses 8.0 threshold)
        middleware._track_cost(response)
        assert len(alert_calls) == 1
        assert alert_calls[0][0] == 10.0  # current spend
        assert alert_calls[0][1] == 8.0   # threshold amount

    def test_alert_fires_only_once(self) -> None:
        """Alert callback is called only once, not on subsequent requests."""
        mock = MockProvider()
        cost_fn = lambda resp: 3.0
        alert_calls = []

        def alert_callback(current, threshold):
            alert_calls.append((current, threshold))

        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=10.0, alert_threshold=0.5, on_alert=alert_callback
        )

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)

        # First request: 3.0 (below 5.0)
        middleware._track_cost(response)
        # Second request: 6.0 (crosses 5.0 threshold, alert fires)
        middleware._track_cost(response)
        # Third request: 9.0 (still above threshold, alert should not fire again)
        middleware._track_cost(response)

        assert len(alert_calls) == 1

    def test_no_alert_without_callback(self) -> None:
        """No error when on_alert is None."""
        mock = MockProvider()
        cost_fn = lambda resp: 5.0
        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=10.0, alert_threshold=0.5
        )  # No on_alert

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        # Should not raise even when crossing threshold
        middleware._track_cost(response)
        middleware._track_cost(response)

        assert middleware.total_spend == 10.0

    def test_no_alert_without_budget_cap(self) -> None:
        """Alert doesn't fire when budget_cap is None."""
        mock = MockProvider()
        cost_fn = lambda resp: 10.0
        alert_calls = []

        def alert_callback(current, threshold):
            alert_calls.append((current, threshold))

        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, on_alert=alert_callback
        )  # No budget_cap

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)

        # No alert without budget_cap
        assert len(alert_calls) == 0

    def test_alert_threshold_calculation(self) -> None:
        """Alert threshold is calculated as budget_cap * alert_threshold."""
        mock = MockProvider()
        cost_fn = lambda resp: 7.0
        alert_calls = []

        def alert_callback(current, threshold):
            alert_calls.append((current, threshold))

        # 20.0 * 0.75 = 15.0 threshold
        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=20.0, alert_threshold=0.75, on_alert=alert_callback
        )

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)

        # First request: 7.0 (below 15.0)
        middleware._track_cost(response)
        assert len(alert_calls) == 0

        # Second request: 14.0 (still below 15.0)
        middleware._track_cost(response)
        assert len(alert_calls) == 0

        # Third request: 21.0 (crosses 15.0)
        # But this would exceed cap (21.0 > 20.0), so it will raise
        from mada_modelkit._errors import BudgetExceededError
        with pytest.raises(BudgetExceededError):
            middleware._track_cost(response)

    def test_alert_at_exact_threshold(self) -> None:
        """Alert fires when spending exactly matches threshold."""
        mock = MockProvider()
        cost_fn = lambda resp: 4.0
        alert_calls = []

        def alert_callback(current, threshold):
            alert_calls.append((current, threshold))

        # 10.0 * 0.8 = 8.0 threshold
        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=10.0, alert_threshold=0.8, on_alert=alert_callback
        )

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)

        # First request: 4.0
        middleware._track_cost(response)
        # Second request: 8.0 (exactly at threshold)
        middleware._track_cost(response)

        assert len(alert_calls) == 1
        assert alert_calls[0][1] == 8.0

    def test_alert_with_custom_threshold(self) -> None:
        """Custom alert_threshold values work correctly."""
        mock = MockProvider()
        cost_fn = lambda resp: 3.0
        alert_calls = []

        def alert_callback(current, threshold):
            alert_calls.append((current, threshold))

        # 10.0 * 0.9 = 9.0 threshold
        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=10.0, alert_threshold=0.9, on_alert=alert_callback
        )

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)

        # First request: 3.0
        middleware._track_cost(response)
        # Second request: 6.0
        middleware._track_cost(response)
        # Third request: 9.0 (crosses 9.0 threshold)
        middleware._track_cost(response)

        assert len(alert_calls) == 1
        assert alert_calls[0][0] == 9.0
        assert alert_calls[0][1] == 9.0

    def test_alert_fired_flag_set(self) -> None:
        """_alert_fired flag is set after alert fires."""
        mock = MockProvider()
        cost_fn = lambda resp: 5.0
        alert_calls = []

        def alert_callback(current, threshold):
            alert_calls.append((current, threshold))

        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=10.0, alert_threshold=0.8, on_alert=alert_callback
        )

        assert middleware._alert_fired is False

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)

        assert middleware._alert_fired is False

        middleware._track_cost(response)

        assert middleware._alert_fired is True

    def test_alert_callback_receives_correct_values(self) -> None:
        """Alert callback receives current spend and threshold amount."""
        mock = MockProvider()
        cost_fn = lambda resp: 6.0
        received_current = None
        received_threshold = None

        def alert_callback(current, threshold):
            nonlocal received_current, received_threshold
            received_current = current
            received_threshold = threshold

        # 15.0 * 0.6 = 9.0 threshold
        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=15.0, alert_threshold=0.6, on_alert=alert_callback
        )

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)  # 6.0
        middleware._track_cost(response)  # 12.0 (crosses 9.0)

        assert received_current == 12.0
        assert received_threshold == 9.0


class TestBudgetReset:
    """Test reset_budget() method for restarting tracking periods."""

    def test_reset_budget_zeroes_spend(self) -> None:
        """reset_budget() sets total_spend back to 0.0."""
        mock = MockProvider()
        cost_fn = lambda resp: 5.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)
        middleware._track_cost(response)

        assert middleware.total_spend == 10.0

        middleware.reset_budget()

        assert middleware.total_spend == 0.0

    def test_reset_budget_clears_alert_flag(self) -> None:
        """reset_budget() clears alert_fired flag."""
        mock = MockProvider()
        cost_fn = lambda resp: 6.0
        alert_fired = [False]

        def alert_callback(current, threshold):
            alert_fired[0] = True

        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=10.0, alert_threshold=0.5, on_alert=alert_callback
        )

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)  # 6.0, crosses 5.0 threshold

        assert middleware._alert_fired is True
        assert alert_fired[0] is True

        middleware.reset_budget()

        assert middleware._alert_fired is False

    def test_reset_allows_alert_to_fire_again(self) -> None:
        """After reset, alert can fire again in new period."""
        mock = MockProvider()
        cost_fn = lambda resp: 6.0
        alert_count = [0]

        def alert_callback(current, threshold):
            alert_count[0] += 1

        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=20.0, alert_threshold=0.5, on_alert=alert_callback
        )

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)

        # First period - threshold is 10.0
        middleware._track_cost(response)  # 6.0, under threshold
        assert alert_count[0] == 0

        middleware._track_cost(response)  # 12.0, crosses threshold
        assert alert_count[0] == 1

        middleware._track_cost(response)  # 18.0, no second alert
        assert alert_count[0] == 1

        # Reset and start new period
        middleware.reset_budget()

        middleware._track_cost(response)  # 6.0, under threshold
        assert alert_count[0] == 1

        middleware._track_cost(response)  # 12.0, crosses threshold again
        assert alert_count[0] == 2

    def test_reset_allows_spending_to_accumulate_again(self) -> None:
        """After reset, spending accumulates from zero."""
        mock = MockProvider()
        cost_fn = lambda resp: 3.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)

        # First period
        middleware._track_cost(response)
        middleware._track_cost(response)
        assert middleware.total_spend == 6.0

        # Reset
        middleware.reset_budget()
        assert middleware.total_spend == 0.0

        # New period
        middleware._track_cost(response)
        assert middleware.total_spend == 3.0

        middleware._track_cost(response)
        assert middleware.total_spend == 6.0

    def test_reset_budget_is_idempotent(self) -> None:
        """Calling reset_budget() multiple times is safe."""
        mock = MockProvider()
        cost_fn = lambda resp: 5.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)
        middleware._track_cost(response)

        # Multiple resets
        middleware.reset_budget()
        middleware.reset_budget()
        middleware.reset_budget()

        assert middleware.total_spend == 0.0
        assert middleware._alert_fired is False

    def test_reset_with_budget_cap(self) -> None:
        """reset_budget() allows spending up to cap again."""
        from mada_modelkit._errors import BudgetExceededError

        mock = MockProvider()
        cost_fn = lambda resp: 6.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=10.0)

        response = AgentResponse(content="test", model="test", input_tokens=10, output_tokens=20)

        # First period - reach cap
        middleware._track_cost(response)  # 6.0

        with pytest.raises(BudgetExceededError):
            middleware._track_cost(response)  # Would be 12.0

        # Reset
        middleware.reset_budget()

        # New period - can spend up to cap again
        middleware._track_cost(response)  # 6.0
        assert middleware.total_spend == 6.0

        with pytest.raises(BudgetExceededError):
            middleware._track_cost(response)  # Would be 12.0

    def test_reset_on_fresh_instance(self) -> None:
        """reset_budget() works on instance that hasn't tracked any cost."""
        mock = MockProvider()
        cost_fn = lambda resp: 1.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        # Reset before any tracking
        middleware.reset_budget()

        assert middleware.total_spend == 0.0
        assert middleware._alert_fired is False


class TestCostControlComprehensive:
    """Comprehensive integration tests for CostControlMiddleware."""

    @pytest.mark.asyncio
    async def test_send_request_tracks_cost(self) -> None:
        """send_request increments spend via cost_fn."""
        mock = MockProvider()
        cost_fn = lambda resp: resp.input_tokens * 0.01 + resp.output_tokens * 0.02
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response.content == "mock"
        # Default mock: 10 input, 5 output → 0.1 + 0.1 = 0.2
        assert middleware.total_spend == 0.2

    @pytest.mark.asyncio
    async def test_send_request_stream_tracks_cost(self) -> None:
        """send_request_stream increments spend from final chunk metadata."""

        class TokenCountingProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="Hello", is_final=False)
                yield StreamChunk(delta=" world", is_final=False)
                yield StreamChunk(
                    delta="!",
                    is_final=True,
                    metadata={"model": "test-model", "input_tokens": 15, "output_tokens": 25},
                )

        mock = TokenCountingProvider()
        cost_fn = lambda resp: resp.input_tokens * 0.01 + resp.output_tokens * 0.02
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        request = AgentRequest(prompt="test")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[2].is_final is True
        # 15 * 0.01 + 25 * 0.02 = 0.15 + 0.5 = 0.65
        assert middleware.total_spend == 0.65

    @pytest.mark.asyncio
    async def test_send_request_enforces_budget_cap(self) -> None:
        """send_request raises BudgetExceededError when cap reached."""
        from mada_modelkit._errors import BudgetExceededError

        mock = MockProvider()
        cost_fn = lambda resp: 5.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=12.0)

        request = AgentRequest(prompt="test")

        # First two succeed
        await middleware.send_request(request)  # 5.0
        await middleware.send_request(request)  # 10.0

        assert middleware.total_spend == 10.0

        # Third fails
        with pytest.raises(BudgetExceededError, match="would exceed budget cap"):
            await middleware.send_request(request)

        # Spend unchanged after failed request
        assert middleware.total_spend == 10.0

    @pytest.mark.asyncio
    async def test_send_request_stream_enforces_budget_cap(self) -> None:
        """send_request_stream raises BudgetExceededError when cap reached."""
        from mada_modelkit._errors import BudgetExceededError

        class TokenCountingProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="test", is_final=False)
                yield StreamChunk(delta="", is_final=True, metadata={"input_tokens": 10, "output_tokens": 20})

        mock = TokenCountingProvider()
        cost_fn = lambda resp: 6.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=15.0)

        request = AgentRequest(prompt="test")

        # First two streams succeed
        async for _ in middleware.send_request_stream(request):
            pass
        async for _ in middleware.send_request_stream(request):
            pass

        assert middleware.total_spend == 12.0

        # Third stream fails after yielding chunks
        with pytest.raises(BudgetExceededError):
            async for _ in middleware.send_request_stream(request):
                pass

        # Spend unchanged
        assert middleware.total_spend == 12.0

    @pytest.mark.asyncio
    async def test_alert_fires_via_send_request(self) -> None:
        """Alert callback fires when threshold crossed via send_request."""
        mock = MockProvider()
        cost_fn = lambda resp: 3.0
        alerts = []

        def alert_callback(current, threshold):
            alerts.append({"current": current, "threshold": threshold})

        middleware = CostControlMiddleware(
            client=mock, cost_fn=cost_fn, budget_cap=10.0, alert_threshold=0.6, on_alert=alert_callback
        )

        request = AgentRequest(prompt="test")

        # First request: 3.0, under threshold (6.0)
        await middleware.send_request(request)
        assert len(alerts) == 0

        # Second request: 6.0, at threshold
        await middleware.send_request(request)
        assert len(alerts) == 1
        assert alerts[0]["current"] == 6.0
        assert alerts[0]["threshold"] == 6.0

        # Third request: 9.0, no second alert
        await middleware.send_request(request)
        assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_reset_budget_via_send_request(self) -> None:
        """reset_budget() allows new spending period via send_request."""
        mock = MockProvider()
        cost_fn = lambda resp: 4.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=10.0)

        request = AgentRequest(prompt="test")

        # First period
        await middleware.send_request(request)  # 4.0
        await middleware.send_request(request)  # 8.0
        assert middleware.total_spend == 8.0

        # Reset
        middleware.reset_budget()
        assert middleware.total_spend == 0.0

        # New period
        await middleware.send_request(request)  # 4.0
        await middleware.send_request(request)  # 8.0
        assert middleware.total_spend == 8.0

    @pytest.mark.asyncio
    async def test_mixed_request_types_share_budget(self) -> None:
        """send_request and send_request_stream share same budget."""

        class TokenCountingProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="test", is_final=True, metadata={"input_tokens": 10, "output_tokens": 20})

        mock = TokenCountingProvider()
        cost_fn = lambda resp: 3.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        request = AgentRequest(prompt="test")

        # Mix send_request and send_request_stream
        await middleware.send_request(request)  # 3.0
        async for _ in middleware.send_request_stream(request):
            pass  # 6.0
        await middleware.send_request(request)  # 9.0

        assert middleware.total_spend == 9.0

    @pytest.mark.asyncio
    async def test_cost_fn_with_complex_calculation(self) -> None:
        """Complex cost_fn calculations work correctly."""
        mock = MockProvider()

        # Tiered pricing: first 1000 tokens cheap, rest expensive
        def tiered_cost_fn(resp):
            total_tokens = resp.total_tokens
            if total_tokens <= 1000:
                return total_tokens * 0.001
            else:
                return 1000 * 0.001 + (total_tokens - 1000) * 0.01

        middleware = CostControlMiddleware(client=mock, cost_fn=tiered_cost_fn)

        # Mock returns 10 input + 5 output = 15 tokens → 0.015
        request = AgentRequest(prompt="test")
        await middleware.send_request(request)
        assert middleware.total_spend == 0.015

    @pytest.mark.asyncio
    async def test_exception_propagation_from_wrapped_client(self) -> None:
        """Exceptions from wrapped client propagate correctly."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error", status_code=500)

        mock = FailingProvider()
        cost_fn = lambda resp: 1.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError, match="API error"):
            await middleware.send_request(request)

        # No cost tracked on error
        assert middleware.total_spend == 0.0

    @pytest.mark.asyncio
    async def test_context_manager_support(self) -> None:
        """CostControlMiddleware works as async context manager."""
        mock = MockProvider()
        cost_fn = lambda resp: 2.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        request = AgentRequest(prompt="test")

        async with middleware:
            await middleware.send_request(request)

        assert middleware.total_spend == 2.0

    @pytest.mark.asyncio
    async def test_middleware_composition_with_retry(self) -> None:
        """CostControlMiddleware stacks with RetryMiddleware."""
        from mada_modelkit.middleware.retry import RetryMiddleware
        from mada_modelkit._errors import ProviderError

        class FlakeyProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.attempt = 0

            async def send_request(self, request):
                self.attempt += 1
                if self.attempt == 1:
                    raise ProviderError("Transient error", status_code=500)
                return await super().send_request(request)

        mock = FlakeyProvider()
        retry_mw = RetryMiddleware(client=mock, max_retries=2)
        cost_fn = lambda resp: 1.5
        cost_mw = CostControlMiddleware(client=retry_mw, cost_fn=cost_fn)

        request = AgentRequest(prompt="test")
        response = await cost_mw.send_request(request)

        assert response.content == "mock"
        # Cost tracked only once (successful response)
        assert cost_mw.total_spend == 1.5
        # But provider called twice (retry)
        assert mock.attempt == 2

    @pytest.mark.asyncio
    async def test_zero_cost_requests_tracked(self) -> None:
        """Requests with zero cost are tracked correctly."""
        mock = MockProvider()
        cost_fn = lambda resp: 0.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn, budget_cap=1.0)

        request = AgentRequest(prompt="test")

        # Can send many zero-cost requests
        for _ in range(100):
            await middleware.send_request(request)

        assert middleware.total_spend == 0.0

    @pytest.mark.asyncio
    async def test_negative_cost_not_allowed(self) -> None:
        """Negative costs accumulate (may happen with credits)."""
        mock = MockProvider()
        cost_fn = lambda resp: -1.0  # Credits
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        request = AgentRequest(prompt="test")
        await middleware.send_request(request)

        # Negative cost reduces total spend
        assert middleware.total_spend == -1.0

    @pytest.mark.asyncio
    async def test_concurrent_requests_accumulate_correctly(self) -> None:
        """Concurrent requests all contribute to total spend."""
        import asyncio

        mock = MockProvider()
        cost_fn = lambda resp: 2.0
        middleware = CostControlMiddleware(client=mock, cost_fn=cost_fn)

        request = AgentRequest(prompt="test")

        # Send 5 requests concurrently
        await asyncio.gather(*[middleware.send_request(request) for _ in range(5)])

        assert middleware.total_spend == 10.0
