from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import pickle
import uuid
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
import time
from typing import Any

import numpy as np

from openpi.rlt import RLTInferDebug
from openpi.rlt import RLTInferRequest
from openpi.rlt import RLTInferResponse
from openpi.rlt import RLTServerRuntimeContract
from openpi.rlt import RLTStatusResponse
from openpi.rlt import RLTTokenDebug
from openpi.rlt import RLTTokenRequest
from openpi.rlt import RLTTokenResponse

from .rlt_feature_model import BaseRLTFeatureModel
from .rlt_feature_model import DeferredJaxRLTFeatureModel
from .rlt_feature_model import RLTFeatureModelObservation

RLT_INFER_ENDPOINT = "/rlt/infer"
RLT_TOKEN_ENDPOINT = "/rlt/token"
RLT_STATUS_ENDPOINT = "/rlt/status"


@dataclass(frozen=True)
class RLTFeatureServer:
    feature_model: BaseRLTFeatureModel
    status: str = "ok"
    runtime_contract: RLTServerRuntimeContract = field(
        default_factory=RLTServerRuntimeContract
    )
    clock: Callable[[], float] = time.perf_counter
    feature_id_factory: Callable[[], str] = field(
        default_factory=lambda: lambda: f"feature-{uuid.uuid4().hex}"
    )

    def handle_infer(self, payload: Mapping[str, Any] | RLTInferRequest) -> RLTInferResponse:
        request = self._coerce_infer_request(payload)
        started_at = self.clock()
        observation = RLTFeatureModelObservation.from_contract(request.observation)
        model_output = self.feature_model.infer_features(
            observation,
            diffusion_steps=request.diffusion_steps,
            reference_horizon=request.reference_horizon,
        )
        reference_plan_norm, reference_plan_list = model_output.require_reference_plan()
        elapsed_s = max(0.0, self.clock() - started_at)
        return RLTInferResponse(
            request_id=request.request_id,
            feature_id=self.feature_id_factory(),
            rl_token=model_output.rl_token.tolist(),
            reference_plan_norm=reference_plan_norm.tolist(),
            reference_plan_list=reference_plan_list.tolist(),
            prefix_valid=model_output.prefix_valid,
            server_infer_time_s=elapsed_s,
            debug=RLTInferDebug(
                rl_token_norm=_l2_norm(model_output.rl_token),
                reference_plan_norm=_l2_norm(reference_plan_norm),
                feature_shape=list(model_output.rl_token.shape),
                reference_plan_shape=list(reference_plan_norm.shape),
            ),
        )

    def handle_token(self, payload: Mapping[str, Any] | RLTTokenRequest) -> RLTTokenResponse:
        request = self._coerce_token_request(payload)
        started_at = self.clock()
        observation = RLTFeatureModelObservation.from_contract(request.observation)
        model_output = self.feature_model.infer_token(observation)
        elapsed_s = max(0.0, self.clock() - started_at)
        return RLTTokenResponse(
            request_id=request.request_id,
            feature_id=self.feature_id_factory(),
            rl_token=model_output.rl_token.tolist(),
            prefix_valid=model_output.prefix_valid,
            server_token_time_s=elapsed_s,
            debug=RLTTokenDebug(
                rl_token_norm=_l2_norm(model_output.rl_token),
                feature_shape=list(model_output.rl_token.shape),
            ),
        )

    def handle_status(self) -> RLTStatusResponse:
        return RLTStatusResponse(status=self.status, server=self.runtime_contract)

    def build_route_handlers(self) -> dict[str, Callable[[Mapping[str, Any] | None], dict[str, Any]]]:
        return {
            RLT_INFER_ENDPOINT: self._dispatch_infer,
            RLT_TOKEN_ENDPOINT: self._dispatch_token,
            RLT_STATUS_ENDPOINT: lambda payload: self._handle_status_payload(payload),
        }

    def dispatch(self, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        handlers = self.build_route_handlers()
        try:
            handler = handlers[path]
        except KeyError as exc:
            raise KeyError(f"unknown RLT feature route: {path}") from exc
        return handler(payload)

    def _handle_status_payload(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if payload not in (None, {}):
            raise ValueError(f"{RLT_STATUS_ENDPOINT} does not accept a request payload")
        return self.handle_status().to_dict()

    def _dispatch_infer(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        return self.handle_infer(
            _expect_payload(payload, path=RLT_INFER_ENDPOINT)
        ).to_dict()

    def _dispatch_token(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        return self.handle_token(
            _expect_payload(payload, path=RLT_TOKEN_ENDPOINT)
        ).to_dict()

    @staticmethod
    def _coerce_infer_request(payload: Mapping[str, Any] | RLTInferRequest) -> RLTInferRequest:
        if isinstance(payload, RLTInferRequest):
            return payload
        return RLTInferRequest.from_dict(payload)

    @staticmethod
    def _coerce_token_request(payload: Mapping[str, Any] | RLTTokenRequest) -> RLTTokenRequest:
        if isinstance(payload, RLTTokenRequest):
            return payload
        return RLTTokenRequest.from_dict(payload)


def _expect_payload(payload: Mapping[str, Any] | None, *, path: str) -> Mapping[str, Any]:
    if payload is None:
        raise ValueError(f"{path} requires a request payload")
    return payload


def _l2_norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float32)))


def make_http_handler(feature_server: RLTFeatureServer) -> type[BaseHTTPRequestHandler]:
    class RLTFeatureRequestHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length) if content_length > 0 else b""
                payload = pickle.loads(body) if body else None
                response = feature_server.dispatch(self.path, payload)
                encoded = pickle.dumps(response)
            except Exception as exc:
                encoded = pickle.dumps({"error": str(exc), "error_type": type(exc).__name__})
                self.send_response(500)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return RLTFeatureRequestHandler


def serve_http(feature_server: RLTFeatureServer, *, host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, int(port)), make_http_handler(feature_server))
    print(f"[rlt_feature_server] listening on {host}:{port}")
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve RLT Stage 2 VLA features over pickle HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    model = DeferredJaxRLTFeatureModel(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        seed=args.seed,
    )
    serve_http(RLTFeatureServer(feature_model=model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
