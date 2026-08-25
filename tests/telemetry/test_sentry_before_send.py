"""Tests for the shared Sentry before_send filter.

default_before_send runs in every entrypoint (API + deriver). It drops known
non-actionable exceptions and collapses DB connection-pool checkout timeouts
into a single warning-level issue so they stop spawning a fresh error issue per
transaction (fleet-wide saturation symptom, tracked in DEV-1852).
"""

from typing import TYPE_CHECKING, Any, cast

import pytest
import sentry_sdk
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from src.exceptions import ResourceNotFoundException
from src.telemetry.sentry import default_before_send, initialize_sentry

if TYPE_CHECKING:
    from sentry_sdk._types import Event, Hint


def _hint(exc: BaseException) -> "Hint":
    return cast("Hint", {"exc_info": (type(exc), exc, None)})


def _event(**kwargs: object) -> "Event":
    return cast("Event", cast(object, dict(kwargs)))


def test_connection_timeout_is_consolidated_and_downgraded() -> None:
    exc = OperationalError("SELECT 1", {}, Exception("connection timeout expired"))
    out = default_before_send({}, _hint(exc))
    assert out == {
        "fingerprint": ["honcho-db-connection-timeout"],
        "level": "warning",
    }


def test_unrelated_operational_error_passes_through() -> None:
    exc = OperationalError("SELECT 1", {}, Exception("some other db failure"))
    event = _event(level="error")
    assert default_before_send(event, _hint(exc)) == {"level": "error"}


def test_honcho_and_validation_errors_are_dropped() -> None:
    assert default_before_send({}, _hint(ResourceNotFoundException("nope"))) is None
    assert (
        default_before_send({}, _hint(ValidationError.from_exception_data("x", [])))
        is None
    )
    assert default_before_send({}, _hint(RequestValidationError([]))) is None


def test_events_without_exc_info_pass_through() -> None:
    event = _event(release="1.0")
    assert default_before_send(event, None) == {"release": "1.0"}
    assert default_before_send(event, cast("Hint", {})) == {"release": "1.0"}


def _captured_before_send(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> object:
    captured: dict[str, object] = {}

    def fake_init(**init_kwargs: object) -> None:
        captured.update(init_kwargs)

    monkeypatch.setattr(sentry_sdk, "init", fake_init)
    initialize_sentry(integrations=[], **kwargs)  # pyright: ignore[reportArgumentType]
    return captured["before_send"]


def test_initialize_sentry_defaults_to_shared_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _captured_before_send(monkeypatch) is default_before_send


def test_initialize_sentry_explicit_none_bypasses_shared_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _captured_before_send(monkeypatch, before_send=None) is None


class TestShutdownSuppression:
    """Teardown failures are expected, not defects, and their volume scales with
    how often the process restarts."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        import src.telemetry.sentry as sentry_module

        original = sentry_module._shutting_down  # pyright: ignore[reportPrivateUsage]
        yield
        sentry_module._shutting_down = original  # pyright: ignore[reportPrivateUsage]

    def test_events_dropped_after_shutdown(self):
        from src.telemetry.sentry import mark_shutting_down

        exc = OperationalError("SELECT 1", {}, Exception("boom"))
        assert default_before_send(_event(), _hint(exc)) is not None

        mark_shutting_down()

        assert default_before_send(_event(), _hint(exc)) is None

    def test_events_without_exc_info_also_dropped(self):
        """capture_message and the logging integration carry no exc_info."""
        from src.telemetry.sentry import mark_shutting_down

        assert default_before_send(_event(), None) is not None
        mark_shutting_down()
        assert default_before_send(_event(), None) is None

    def test_logging_integration_events_are_also_dropped(self):
        """logger.error() raises its own event, separate from capture_exception.

        Teardown paths log before they capture, so filtering only the explicit
        capture would let the logged copy through.
        """
        import logging

        import sentry_sdk
        from sentry_sdk.transport import Transport

        from src.telemetry.sentry import mark_shutting_down

        sent: list[str] = []

        class RecordingTransport(Transport):
            def capture_envelope(self, envelope: Any) -> None:
                sent.extend(item.type for item in envelope.items)

        sentry_sdk.init(
            dsn="https://public@example.ingest.sentry.io/1",
            before_send=default_before_send,
            transport=RecordingTransport(),
        )
        log = logging.getLogger("test_shutdown_suppression")

        def log_an_error() -> None:
            try:
                raise RuntimeError("connection reset during query")
            except RuntimeError:
                log.error("query failed", exc_info=True)

        log_an_error()
        sentry_sdk.flush()
        assert "event" in sent, "sanity: ERROR logs reach Sentry normally"

        sent.clear()
        mark_shutting_down()

        log_an_error()
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            sentry_sdk.capture_exception(exc)
        sentry_sdk.flush()

        assert "event" not in sent, "logged and captured copies are both dropped"
