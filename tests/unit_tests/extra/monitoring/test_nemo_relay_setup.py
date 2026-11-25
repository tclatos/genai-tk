from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from genai_tk.extra.monitoring import nemo_relay_setup


def test_relay_otlp_allows_explicit_http_endpoint(monkeypatch) -> None:
    config = MagicMock()
    config.allow_insecure = False
    config.set_header = MagicMock()
    subscriber = MagicMock()
    nemo_relay = SimpleNamespace(
        OpenTelemetryConfig=MagicMock(return_value=config),
        OpenTelemetrySubscriber=MagicMock(return_value=subscriber),
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://host.openshell.internal:4318")

    with patch.dict("sys.modules", {"nemo_relay": nemo_relay}):
        nemo_relay_setup._setup_relay_otlp(
            enable_otlp=True,
            otlp_endpoint="http://host.openshell.internal:4318/v1/traces",
        )

    assert config.allow_insecure is True
    subscriber.register.assert_called_once()
