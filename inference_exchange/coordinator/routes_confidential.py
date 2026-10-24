"""Confidential inference relay — coordinator never sees plaintext.

POST /v1/confidential/infer accepts an encrypted envelope and relays it
unchanged to the provider. The coordinator routes by session metadata only.

Security invariant: this endpoint MUST reject any request containing a
``messages`` field. The coordinator MUST NOT decrypt, parse, or inspect
the ``encrypted_envelope``.
"""

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from inference_exchange.shared.errors import (
    NoProviderAvailable,
    ProviderError,
    ProviderTimeout,
    RateLimitExceeded,
)
from inference_exchange.shared.protocol import (
    InferenceDone,
    InferenceError,
    InferenceRequest,
    InferenceResponseChunk,
)

from .dependencies import (
    _add_trace,
    _rate_limiter,
    get_auth,
    get_billing,
    get_hub,
    get_reputation_tracker,
    get_tps_tracker,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _charge_confidential(
    request_id: str,
    request: "ConfidentialInferenceRequest",
    provider,
    consumer_id: str,
    output_tokens: int,
):
    """Charge the encrypted relay for the output delivered to the consumer."""
    get_billing().charge_request(
        request_id=request_id,
        consumer_id=consumer_id,
        provider_id=provider.provider_id,
        model=request.model,
        input_tokens=request.estimated_input_tokens,
        output_tokens=output_tokens,
        price_per_mtok_input=provider.capabilities.price_per_mtok_input,
        price_per_mtok_output=provider.capabilities.price_per_mtok_output,
    )


class ConfidentialInferenceRequest(BaseModel):
    """Request body for the confidential relay. No ``messages`` field."""

    session_id: str
    encrypted_envelope: dict
    model: str = "default"
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = True
    estimated_input_tokens: int = 0
    sequence_number: int = 0

    # Plaintext messages are forbidden on the confidential path.
    messages: list | None = None


@router.post("/v1/confidential/infer")
async def confidential_infer(request: ConfidentialInferenceRequest, raw_request: Request):
    """Relay an encrypted inference request to a provider.

    The coordinator sees: session_id, model, estimated_input_tokens, and the
    opaque encrypted_envelope. It does NOT see prompt or response content.
    """
    # CB-1: reject any request that contains plaintext messages
    if request.messages is not None:
        return JSONResponse(
            {"error": {"type": "plaintext_rejected",
                       "message": "The confidential endpoint does not accept a messages field. "
                                  "Encrypt messages client-side and send encrypted_envelope."}},
            status_code=400,
        )

    hub = get_hub()
    auth = get_auth()

    # Authenticate consumer
    from .routes_auth import resolve_user_from_request
    user_info = resolve_user_from_request(raw_request)
    consumer_id = user_info["user_id"] if user_info else auth.resolve_consumer(
        raw_request.headers.get("authorization"))

    from .routes_auth import anonymous_inference_blocked, sign_in_required_response
    if anonymous_inference_blocked(consumer_id):
        return sign_in_required_response()

    # Rate limit
    if not _rate_limiter.allow(consumer_id):
        raise RateLimitExceeded()

    # Balance check
    from .dependencies import get_store
    store = get_store()
    account = store.get_account(consumer_id)
    if account and account["balance_micro"] <= 0:
        return JSONResponse(
            {"error": {"type": "insufficient_balance",
                       "message": "Insufficient balance. Add credits to continue."}},
            status_code=402,
        )

    request_id = str(uuid.uuid4())

    # Select a provider (the hub filters out un-admitted confidential providers)
    provider = hub.select_provider(
        request.model,
        preference="balanced",
        session_id=request.session_id,
        reputation_fn=get_reputation_tracker().get_score,
    )

    if provider is None:
        _add_trace({
            "request_id": request_id[:8],
            "timestamp": time.time(),
            "model": request.model,
            "status": "no_provider",
            "confidential": True,
        }, consumer_id)
        raise NoProviderAvailable()

    # CB-3: build an InferenceRequest with NO plaintext — relay the envelope as-is
    inference_req = InferenceRequest(
        request_id=request_id,
        model=request.model,
        messages=None,
        encrypted_body=request.encrypted_envelope,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream,
    )

    queue = hub.create_response_queue(request_id)
    try:
        await hub.send_to_provider(provider, inference_req)
    except Exception as e:
        hub.remove_response_queue(request_id)
        raise ProviderError(f"Failed to reach provider: {e}")

    _add_trace({
        "request_id": request_id[:8],
        "timestamp": time.time(),
        "model": request.model,
        "status": "matched",
        "confidential": True,
        "selected_provider": provider.name,
        "encrypted": True,
    }, consumer_id)

    if request.stream:
        return StreamingResponse(
            _stream_confidential(request_id, request, queue, hub, provider, consumer_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-OCIP-Provider": provider.name,
                "X-OCIP-Confidential": "true",
            },
        )
    else:
        return await _collect_confidential(request_id, request, queue, hub, provider, consumer_id)


async def _stream_confidential(
    request_id: str, request: ConfidentialInferenceRequest,
    queue: asyncio.Queue, hub, provider, consumer_id: str,
):
    """Stream encrypted response chunks. CB-4: relay without decrypting."""
    token_count = 0
    start_time = time.time()
    outcome = "success"
    completed = False
    billed = False
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                outcome = "timeout"
                yield f"data: {json.dumps({'error': {'type': 'provider_timeout'}})}\n\n"
                break

            if isinstance(msg, InferenceResponseChunk):
                token_count += 1
                chunk = {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": msg.finish_reason}],
                }
                # CB-4: pass through encrypted_token unchanged
                if msg.encrypted_token:
                    chunk["ocip_encrypted_token"] = msg.encrypted_token
                yield f"data: {json.dumps(chunk)}\n\n"

            elif isinstance(msg, InferenceDone):
                completed = True
                chunk = {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                break

            elif isinstance(msg, InferenceError):
                completed = True
                outcome = "error"
                yield f"data: {json.dumps({'error': {'type': 'provider_error', 'message': msg.error}})}\n\n"
                break

        yield "data: [DONE]\n\n"

        # Bill using estimated input tokens (coordinator can't count from ciphertext)
        # and provider-reported output tokens (FI-4)
        _charge_confidential(request_id, request, provider, consumer_id, token_count)
        billed = True

        elapsed = time.time() - start_time
        if token_count > 0 and elapsed > 0:
            get_tps_tracker().record_request(
                provider_id=provider.provider_id, model=request.model,
                tokens=token_count, seconds=elapsed,
                hardware=provider.capabilities.hardware,
            )

        reputation = get_reputation_tracker()
        elapsed_ms = int((time.time() - start_time) * 1000)
        if outcome == "success":
            reputation.record_success(provider.provider_id, tokens=token_count, latency_ms=elapsed_ms)
        elif outcome == "timeout":
            reputation.record_timeout(provider.provider_id)
        else:
            reputation.record_error(provider.provider_id)
    finally:
        if not completed:
            try:
                await hub.cancel_request(request_id, output_tokens=token_count)
            except Exception as e:
                logger.warning(f"[{request_id[:8]}] Failed to cancel provider request: {e}")
            if not billed:
                _charge_confidential(request_id, request, provider, consumer_id, token_count)
        hub.remove_response_queue(request_id)


async def _collect_confidential(
    request_id: str, request: ConfidentialInferenceRequest,
    queue: asyncio.Queue, hub, provider, consumer_id: str,
) -> dict:
    """Collect all encrypted tokens into a single non-streaming response."""
    encrypted_tokens: list[dict] = []
    start_time = time.time()
    completed = False
    errored = False
    token_count = 0
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                get_reputation_tracker().record_timeout(provider.provider_id)
                raise ProviderTimeout()

            if isinstance(msg, InferenceResponseChunk):
                if msg.encrypted_token:
                    encrypted_tokens.append(msg.encrypted_token)
            elif isinstance(msg, InferenceDone):
                completed = True
                break
            elif isinstance(msg, InferenceError):
                completed = True
                errored = True
                get_reputation_tracker().record_error(provider.provider_id)
                raise ProviderError(msg.error)
    finally:
        token_count = len(encrypted_tokens)
        if not completed:
            try:
                await hub.cancel_request(request_id, output_tokens=token_count)
            except Exception as e:
                logger.warning(f"[{request_id[:8]}] Failed to cancel provider request: {e}")
        if not completed or errored:
            _charge_confidential(request_id, request, provider, consumer_id, token_count)
        hub.remove_response_queue(request_id)

    token_count = len(encrypted_tokens)
    _charge_confidential(request_id, request, provider, consumer_id, token_count)

    elapsed = time.time() - start_time
    if token_count > 0 and elapsed > 0:
        get_tps_tracker().record_request(
            provider_id=provider.provider_id, model=request.model,
            tokens=token_count, seconds=elapsed,
            hardware=provider.capabilities.hardware,
        )
    get_reputation_tracker().record_success(
        provider.provider_id, tokens=token_count, latency_ms=int(elapsed * 1000))

    return {
        "id": f"chatcmpl-{request_id[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "ocip_encrypted_tokens": encrypted_tokens,
        "usage": {
            "prompt_tokens": request.estimated_input_tokens,
            "completion_tokens": token_count,
            "total_tokens": request.estimated_input_tokens + token_count,
        },
    }
