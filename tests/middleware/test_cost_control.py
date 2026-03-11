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
