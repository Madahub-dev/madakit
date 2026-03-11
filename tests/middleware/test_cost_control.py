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
