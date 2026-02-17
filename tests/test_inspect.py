from __future__ import annotations

"""
Unit tests for:
  - models.StageTraceResponse / InspectResponse (Pydantic contracts)
  - api.inspect_router._trace_to_response() (internal adapter)
  - api.inspect_router.inspect endpoint (via TestClient)
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf, DictConfig
from pydantic import ValidationError

from src.api.app import create_app
from src.api.inspect_router import _stage_trace_to_response, _trace_to_response
from src.core import PipelineTrace, StageTrace
from src.config import RESOURCES
from src.models import (
    FinalResponse,
    InspectResponse,
    Label,
    ResponseCode,
    StageTraceResponse,
)
from src.models.response import ResponseData
from tests.conftest import make_stage_result

DET_TARGET = "src.stages.deterministic.DeterministicStage"


def _det_cfg() -> dict:
    return {
        "_target_": DET_TARGET,
        "keywords_crisis_path": str(RESOURCES / "keywords_crisis.yaml"),
        "keywords_malign_path": str(RESOURCES / "keywords_malign.yaml"),
        "bypass_patterns_path": str(RESOURCES / "bypass_patterns.yaml"),
    }


def _stages_cfg(*stage_dicts: dict) -> DictConfig:
    return OmegaConf.create({"stages": list(stage_dicts)})


def _pipeline_cfg(*stage_dicts: dict) -> DictConfig:
    pipeline = OmegaConf.create({"pipeline": _stages_cfg(*stage_dicts)})
    return pipeline


def _make_final_response(label: Label = Label.VALID) -> FinalResponse:
    from src.models import LABEL_TO_CODE  # noqa: PLC0415
    return FinalResponse(
        code=LABEL_TO_CODE[label],
        label=label,
        data=ResponseData(
            processed_text="test prompt",
            confidence_score=0.9,
            metadata={"stage": "deterministic"},
        ),
    )


def _make_stage_trace(
        label: Label = Label.VALID,
        duration_ms: float = 5.0,
        short_circuit: bool = False,
        triggered_by: str | None = None,
) -> StageTrace:
    result = make_stage_result(
        label,
        stage_name="deterministic",
        confidence=0.9,
        triggered_by=triggered_by,
        short_circuit=short_circuit,
    )
    return StageTrace(result=result, duration_ms=duration_ms)


def _make_pipeline_trace(
        stage_traces: tuple[StageTrace, ...] = (),
        all_stage_names: tuple[str, ...] | None = None,
        total_duration_ms: float = 10.0,
        label: Label = Label.VALID,
) -> PipelineTrace:
    if all_stage_names is None:
        # Default: all_stage_names = names of executed stages
        all_stage_names = tuple(st.result.stage_name for st in stage_traces)
    return PipelineTrace(
        stage_traces=stage_traces,
        all_stage_names=all_stage_names,
        total_duration_ms=total_duration_ms,
        final_response=_make_final_response(label),
    )


# ---------------------------------------------------------------------------
# StageTraceResponse — Pydantic model
# ---------------------------------------------------------------------------

class TestStageTraceResponse:
    def test_valid_construction(self) -> None:
        r = StageTraceResponse(
            stage="deterministic",
            label=Label.VALID,
            score=0.9,
            triggered_by=None,
            short_circuit=False,
            duration_ms=3.2,
        )
        assert r.stage == "deterministic"
        assert r.label == Label.VALID
        assert r.score == pytest.approx(0.9)
        assert r.duration_ms == pytest.approx(3.2)

    def test_score_must_be_between_0_and_1(self) -> None:
        with pytest.raises(ValidationError):
            StageTraceResponse(
                stage="s",
                label=Label.VALID,
                score=1.5,
                short_circuit=False,
                duration_ms=1.0,
            )

    def test_duration_ms_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            StageTraceResponse(
                stage="s",
                label=Label.VALID,
                score=0.9,
                short_circuit=False,
                duration_ms=-1.0,
            )

    def test_triggered_by_is_optional(self) -> None:
        r = StageTraceResponse(
            stage="s", label=Label.VALID, score=0.5,
            short_circuit=False, duration_ms=1.0,
        )
        assert r.triggered_by is None

    def test_all_labels_accepted(self) -> None:
        for label in Label:
            r = StageTraceResponse(
                stage="s", label=label, score=0.5,
                short_circuit=False, duration_ms=1.0,
            )
            assert r.label == label


# ---------------------------------------------------------------------------
# InspectResponse — Pydantic model
# ---------------------------------------------------------------------------

class TestInspectResponse:
    def test_valid_construction(self) -> None:
        r = InspectResponse(
            final=_make_final_response(),
            trace=[],
            skipped_stages=[],
            total_stages=1,
            total_duration_ms=10.0,
        )
        assert r.total_duration_ms == pytest.approx(10.0)
        assert r.trace == []
        assert r.skipped_stages == []
        assert r.total_stages == 1

    def test_total_duration_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            InspectResponse(
                final=_make_final_response(),
                trace=[],
                skipped_stages=[],
                total_stages=1,
                total_duration_ms=-5.0,
            )

    def test_trace_preserves_order(self) -> None:
        stages = [
            StageTraceResponse(stage=f"s{i}", label=Label.VALID,
                               score=0.5, short_circuit=False, duration_ms=1.0)
            for i in range(3)
        ]
        r = InspectResponse(
            final=_make_final_response(),
            trace=stages,
            skipped_stages=[],
            total_stages=3,
            total_duration_ms=5.0,
        )
        assert [s.stage for s in r.trace] == ["s0", "s1", "s2"]

    def test_json_serialization_roundtrip(self) -> None:
        r = InspectResponse(
            final=_make_final_response(Label.CRISIS),
            trace=[StageTraceResponse(
                stage="deterministic", label=Label.CRISIS,
                score=1.0, short_circuit=True, duration_ms=2.1,
                triggered_by="keyword:suicidio",
            )],
            skipped_stages=["semantic_bert"],
            total_stages=2,
            total_duration_ms=2.4,
        )
        data = r.model_dump(mode="json")
        assert data["final"]["label"] == "Crisis"
        assert data["trace"][0]["short_circuit"] is True
        assert data["trace"][0]["triggered_by"] == "keyword:suicidio"
        assert data["skipped_stages"] == ["semantic_bert"]
        assert data["total_stages"] == 2
        assert data["total_duration_ms"] == pytest.approx(2.4)


# ---------------------------------------------------------------------------
# _trace_to_response / _stage_trace_to_response — adapter functions
# ---------------------------------------------------------------------------

class TestAdapter:
    def test_stage_trace_to_response_maps_fields(self) -> None:
        st = _make_stage_trace(label=Label.CRISIS, duration_ms=7.5,
                               short_circuit=True, triggered_by="kw:test")
        r = _stage_trace_to_response(st)

        assert r.stage == "deterministic"
        assert r.label == Label.CRISIS
        assert r.score == pytest.approx(0.9)
        assert r.duration_ms == pytest.approx(7.5)
        assert r.short_circuit is True
        assert r.triggered_by == "kw:test"

    def test_trace_to_response_empty_trace(self) -> None:
        pt = _make_pipeline_trace(stage_traces=(), total_duration_ms=1.0)
        r = _trace_to_response(pt)

        assert r.trace == []
        assert r.total_duration_ms == pytest.approx(1.0)

    def test_trace_to_response_preserves_stage_count(self) -> None:
        traces = tuple(_make_stage_trace() for _ in range(3))
        pt = _make_pipeline_trace(stage_traces=traces, total_duration_ms=15.0)
        r = _trace_to_response(pt)

        assert len(r.trace) == 3

    def test_trace_to_response_final_matches_pipeline(self) -> None:
        pt = _make_pipeline_trace(label=Label.MALIGN, total_duration_ms=5.0)
        r = _trace_to_response(pt)

        assert r.final.label == Label.MALIGN
        assert r.final.code == ResponseCode.MALIGN

    def test_trace_to_response_total_duration_preserved(self) -> None:
        pt = _make_pipeline_trace(total_duration_ms=42.123)
        r = _trace_to_response(pt)

        assert r.total_duration_ms == pytest.approx(42.123)

    def test_skipped_stages_when_short_circuit(self) -> None:
        executed = (_make_stage_trace(label=Label.CRISIS, short_circuit=True),)
        pt = _make_pipeline_trace(
            stage_traces=executed,
            all_stage_names=("deterministic", "semantic_bert", "attack_detection"),
        )
        r = _trace_to_response(pt)

        assert r.skipped_stages == ["semantic_bert", "attack_detection"]
        assert r.total_stages == 3

    def test_no_skipped_stages_when_all_ran(self) -> None:
        executed = (
            _make_stage_trace(label=Label.VALID, short_circuit=False),
            _make_stage_trace(label=Label.VALID, short_circuit=False),
        )
        # Assign distinct stage names
        from src.core import StageTrace  # noqa: PLC0415
        sr1 = make_stage_result(Label.VALID, stage_name="s1", short_circuit=False)
        sr2 = make_stage_result(Label.VALID, stage_name="s2", short_circuit=False)
        t1 = StageTrace(result=sr1, duration_ms=1.0)
        t2 = StageTrace(result=sr2, duration_ms=1.0)
        pt = _make_pipeline_trace(
            stage_traces=(t1, t2),
            all_stage_names=("s1", "s2"),
        )
        r = _trace_to_response(pt)

        assert r.skipped_stages == []
        assert r.total_stages == 2

    def test_skipped_stages_order_matches_registration(self) -> None:
        """Skipped stages appear in registration order, not alphabetical."""
        sr = make_stage_result(Label.MALIGN, stage_name="det", short_circuit=True)
        executed = (StageTrace(result=sr, duration_ms=1.0),)
        pt = _make_pipeline_trace(
            stage_traces=executed,
            all_stage_names=("det", "zzz_bert", "aaa_attack"),
        )
        r = _trace_to_response(pt)

        assert r.skipped_stages == ["zzz_bert", "aaa_attack"]


# ---------------------------------------------------------------------------
# /v1/inspect endpoint — integration via TestClient
# ---------------------------------------------------------------------------

class TestInspectEndpoint:
    def _app_with_inspect(self):
        cfg = _pipeline_cfg()
        app = create_app(pipeline_cfg=cfg)
        # Patch env var check done at app creation time
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("SAFEGUARD_INSPECT_MODE", "true")
            # Re-create so the route is registered
            app = create_app(pipeline_cfg=cfg)
        return app

    def test_inspect_endpoint_registered_when_mode_enabled(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(), inspect_mode=True)
        routes = [r.path for r in app.routes]
        assert "/v1/inspect" in routes

    def test_inspect_endpoint_not_registered_by_default(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(), inspect_mode=False)
        routes = [r.path for r in app.routes]
        assert "/v1/inspect" not in routes

    def test_inspect_endpoint_not_registered_when_false(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(), inspect_mode=False)
        routes = [r.path for r in app.routes]
        assert "/v1/inspect" not in routes

    def test_inspect_returns_200(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(_det_cfg()), inspect_mode=True)
        with TestClient(app) as client:
            resp = client.post("/v1/inspect", json={"text": "Hello world"})
        assert resp.status_code == 200

    def test_inspect_response_structure(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(_det_cfg()), inspect_mode=True)
        with TestClient(app) as client:
            resp = client.post("/v1/inspect", json={"text": "Hello world"})
        body = resp.json()

        assert "final" in body
        assert "trace" in body
        assert "skipped_stages" in body
        assert "total_stages" in body
        assert "total_duration_ms" in body
        assert isinstance(body["trace"], list)
        assert isinstance(body["skipped_stages"], list)
        assert body["total_stages"] >= 1
        assert body["total_duration_ms"] >= 0

    def test_inspect_trace_contains_stage_fields(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(_det_cfg()), inspect_mode=True)
        with TestClient(app) as client:
            resp = client.post("/v1/inspect", json={"text": "Hello world"})
        trace = resp.json()["trace"]

        assert len(trace) >= 1
        stage = trace[0]
        assert "stage" in stage
        assert "label" in stage
        assert "score" in stage
        assert "short_circuit" in stage
        assert "duration_ms" in stage

    def test_inspect_crisis_shows_short_circuit(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(_det_cfg()), inspect_mode=True)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/inspect",
                # "wish i was dead" is a confirmed keyword in keywords_crisis.yaml
                json={"text": "I wish i was dead and nobody cares"},
            )
        body = resp.json()

        assert body["final"]["label"] == "Crisis"
        assert body["trace"][0]["short_circuit"] is True
        # Single-stage pipeline: no stages to skip
        assert body["skipped_stages"] == []
        assert body["total_stages"] == 1

    def test_inspect_final_matches_evaluate(self) -> None:
        """The final field of /v1/inspect must match /v1/evaluate for the same input."""
        app = create_app(pipeline_cfg=_pipeline_cfg(_det_cfg()), inspect_mode=True)
        with TestClient(app) as client:
            text = "How do I manage my anxiety?"
            inspect_resp = client.post("/v1/inspect", json={"text": text})
            eval_resp = client.post("/v1/evaluate", json={"text": text})

        assert inspect_resp.json()["final"]["label"] == eval_resp.json()["label"]
        assert inspect_resp.json()["final"]["code"] == eval_resp.json()["code"]

    def test_inspect_rejects_empty_text(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(_det_cfg()), inspect_mode=True)
        with TestClient(app) as client:
            resp = client.post("/v1/inspect", json={"text": ""})
        assert resp.status_code == 422

    def test_inspect_fail_closed_on_pipeline_error(self) -> None:
        """
        The fail-closed guarantee lives inside SafeguardPipeline.evaluate_with_trace().
        If we mock the entire pipeline object, we bypass that guarantee — the mock raises
        directly to the endpoint, which correctly returns 500.

        This test verifies the boundary: the endpoint does NOT swallow arbitrary exceptions
        from a broken pipeline object (that would hide real bugs). The real fail-closed
        path is tested in test_pipeline.py::TestEvaluateWithTrace::test_fail_closed_*.
        """
        app = create_app(pipeline_cfg=_pipeline_cfg(_det_cfg()), inspect_mode=True)

        broken_pipeline = MagicMock()
        broken_pipeline.evaluate_with_trace.side_effect = RuntimeError("forced crash")

        with TestClient(app, raise_server_exceptions=False) as client:
            app.state.safeguard_pipeline = broken_pipeline
            resp = client.post("/v1/inspect", json={"text": "hello"})

        assert resp.status_code == 500
