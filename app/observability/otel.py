"""
Optional OpenTelemetry layer. No-op unless OTel is installed AND enabled via
env (OTEL_TRACES=1 for console export, or set OTEL_EXPORTER_OTLP_ENDPOINT).
Lets the in-house tracer also emit real spans to a backend in production.
"""

import os

_tracer = None
_enabled = False


def _setup():
    global _tracer, _enabled
    if os.getenv("OTEL_TRACES") != "1" and not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        provider = TracerProvider()
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        else:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("coderag")
        _enabled = True
        print("[otel] tracing enabled")
    except Exception as e:  # OTel not installed or misconfigured -> stay no-op
        print(f"[otel] disabled ({type(e).__name__}); in-house tracing still active.")


_setup()


def start_span(name: str):
    if not _enabled or _tracer is None:
        return None
    return _tracer.start_span(name)


def end_span(span) -> None:
    if span is not None:
        try:
            span.end()
        except Exception:
            pass
