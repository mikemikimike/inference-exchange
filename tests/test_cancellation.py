"""Regression tests for cancellation propagation and partial billing."""

import asyncio
import threading

import pytest

from inference_exchange.config import ProviderConfig
from inference_exchange.coordinator.provider_hub import ConnectedProvider, ProviderHub
from inference_exchange.coordinator.routes_confidential import (
    ConfidentialInferenceRequest,
    _stream_confidential,
)
from inference_exchange.coordinator.routes_inference import _stream_response
from inference_exchange.provider.agent import ProviderAgent
from inference_exchange.shared.protocol import (
    InferenceError,
    InferenceRequest,
    ProviderCapabilities,
)


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)

    async def send(self, data):
        self.sent.append(data)


class FakeBilling:
    def __init__(self):
        self.charges = []

    def charge_request(self, **kwargs):
        self.charges.append(kwargs)


def _provider(hub: ProviderHub, request_id: str):
    websocket = FakeWebSocket()
    provider = ConnectedProvider(
        provider_id="provider-1",
        name="test-provider",
        ws=websocket,
        capabilities=ProviderCapabilities(models=["test-model"]),
    )
    hub._providers[provider.provider_id] = provider
    queue = hub.create_response_queue(request_id)
    return provider, websocket, queue


async def _send_request(hub: ProviderHub, provider, request_id: str):
    await hub.send_to_provider(
        provider,
        InferenceRequest(request_id=request_id, model="test-model", messages=[]),
    )


@pytest.mark.asyncio
async def test_stream_cancellation_bills_delivered_tokens():
    hub = ProviderHub()
    provider, websocket, queue = _provider(hub, "cancelled")
    await _send_request(hub, provider, "cancelled")
    billed = []
    hub.register_billing_callback("cancelled", lambda done: billed.append(done.tokens_generated))

    hub.handle_provider_message(
        "provider-1",
        {
            "type": "inference_response",
            "request_id": "cancelled",
            "token": "one",
        },
    )
    stream = _stream_response(
        "cancelled",
        "test-model",
        queue,
        hub,
        provider,
        "consumer",
        0.0,
    )
    chunk = await stream.__anext__()
    assert '"content": "one"' in chunk

    await stream.aclose()

    assert billed == [1]
    assert websocket.sent[-1] == {"type": "cancel_request", "request_id": "cancelled"}
    assert provider.active_requests == 0
    assert "cancelled" not in hub._response_queues


@pytest.mark.asyncio
async def test_provider_error_bills_delivered_tokens():
    hub = ProviderHub()
    provider, _, queue = _provider(hub, "failed")
    await _send_request(hub, provider, "failed")
    billed = []
    hub.register_billing_callback("failed", lambda done: billed.append(done.tokens_generated))

    hub.handle_provider_message(
        "provider-1",
        {
            "type": "inference_response",
            "request_id": "failed",
            "token": "one",
        },
    )
    hub.handle_provider_message(
        "provider-1",
        {
            "type": "inference_error",
            "request_id": "failed",
            "error": "provider failed",
        },
    )
    await queue.get()
    await queue.get()

    assert billed == [1]
    assert provider.active_requests == 0


@pytest.mark.asyncio
async def test_confidential_stream_cancellation_bills_delivered_tokens(monkeypatch):
    hub = ProviderHub()
    provider, websocket, queue = _provider(hub, "confidential")
    await _send_request(hub, provider, "confidential")
    billing = FakeBilling()
    monkeypatch.setattr(
        "inference_exchange.coordinator.routes_confidential.get_billing",
        lambda: billing,
    )
    request = ConfidentialInferenceRequest(
        session_id="session",
        encrypted_envelope={"ciphertext": "opaque"},
        model="test-model",
        estimated_input_tokens=4,
    )

    hub.handle_provider_message(
        "provider-1",
        {
            "type": "inference_response",
            "request_id": "confidential",
            "encrypted_token": {"ciphertext": "token"},
        },
    )
    stream = _stream_confidential(
        "confidential",
        request,
        queue,
        hub,
        provider,
        "consumer",
    )
    chunk = await stream.__anext__()
    assert '"ocip_encrypted_token"' in chunk

    await stream.aclose()

    assert billing.charges[0]["input_tokens"] == 4
    assert billing.charges[0]["output_tokens"] == 1
    assert websocket.sent[-1] == {"type": "cancel_request", "request_id": "confidential"}


@pytest.mark.asyncio
async def test_simple_provider_agent_cancels_active_task():
    released = threading.Event()

    class BlockingEngine:
        model_name = "test-model"

        def generate_stream(self, **_kwargs):
            released.wait(timeout=5)
            yield "finished"

    agent = ProviderAgent(
        ProviderConfig(coordinator_url="ws://unused", provider_name="test"),
        BlockingEngine(),
    )
    websocket = FakeWebSocket()
    request = InferenceRequest(
        request_id="agent-cancelled",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )
    task = asyncio.create_task(agent._handle_inference(websocket, request))
    await asyncio.sleep(0.05)
    task.cancel()
    released.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    errors = [InferenceError.model_validate_json(raw) for raw in websocket.sent]
    assert errors and errors[-1].error == "cancelled"
    assert agent._active_requests == 0
    assert request.request_id not in agent._active_request_tasks
