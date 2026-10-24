"""Inference endpoints — POST /v1/chat/completions (OpenAI-compatible)."""

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference_exchange.shared.crypto import encrypt_json
from inference_exchange.shared.errors import (
    NoProviderAvailable,
    ProviderError,
    ProviderTimeout,
    QueueFull,
    QueueTimeout,
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
    get_event_bus,
    get_hub,
    get_reputation_tracker,
    get_tps_tracker,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request/Response models (OpenAI-compatible) ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = True
    # OCIP routing preferences
    ocip_preference: str = "balanced"
    ocip_min_confidence: str = "hardened"
    ocip_max_price: float | None = None
    ocip_session_id: str | None = None
    ocip_consumer_public_key: str | None = None

    def validate_inputs(self) -> str | None:
        """Return error message if invalid, None if OK."""
        if not self.messages:
            return "Messages cannot be empty"
        if len(self.messages) > 100:
            return "Too many messages (max 100)"
        total_chars = sum(len(m.content) for m in self.messages)
        if total_chars > 200_000:
            return "Total message content too large (max 200k chars)"
        if self.max_tokens < 1 or self.max_tokens > 32768:
            return "max_tokens must be 1-32768"
        if self.temperature < 0 or self.temperature > 2:
            return "temperature must be 0-2"
        return None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "inference-exchange"


def _estimate_input_tokens(messages: list) -> int:
    """Estimate token count from messages (approximate: ~4 chars per token for English).

    This is a fast approximation. For exact counting, use the model's tokenizer.
    Good enough for billing at the micro-USD level.
    """
    total_chars = sum(len(str(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))) for m in messages)
    # Add ~4 tokens per message for role/formatting overhead
    overhead = len(messages) * 4
    return max(1, total_chars // 4 + overhead)


def _make_billing_callback(
    request_id: str,
    model: str,
    provider,
    consumer_id: str,
    input_token_estimate: int,
    start_time: float,
    preference: str = "",
    queued: bool = False,
    queue_wait_ms: int = 0,
):
    """Create a billing callback invoked when InferenceDone arrives from the provider.

    This decouples billing from the streaming generator lifecycle. The WebSocket
    handler calls this as soon as InferenceDone arrives, regardless of whether
    the consumer HTTP response is still open.
    """

    def callback(done_msg: InferenceDone):
        actual_input = done_msg.input_tokens if done_msg.input_tokens > 0 else input_token_estimate
        actual_cached = done_msg.cached_tokens
        output_tokens = done_msg.tokens_generated

        elapsed = time.time() - start_time
        latency_ms = int(elapsed * 1000)
        tps = output_tokens / elapsed if elapsed > 0 and output_tokens > 0 else 0

        billing = get_billing()
        billing.charge_request(
            request_id=request_id,
            consumer_id=consumer_id,
            provider_id=provider.provider_id,
            model=model,
            input_tokens=actual_input,
            output_tokens=output_tokens,
            price_per_mtok_input=provider.capabilities.price_per_mtok_input,
            price_per_mtok_output=provider.capabilities.price_per_mtok_output,
            cached_tokens=actual_cached,
            price_per_mtok_cache=provider.capabilities.price_per_mtok_cache,
            provider_name=provider.name,
            trust_level=provider.capabilities.trust_level.value,
            encrypted=bool(provider.encryption_public_key),
            preference=preference,
            latency_ms=latency_ms,
            tps=round(tps, 2),
            queued=queued,
            queue_wait_ms=queue_wait_ms,
        )

        logger.info(
            f"[{request_id[:8]}] Billed via callback: "
            f"in={actual_input} cached={actual_cached} out={output_tokens} "
            f"latency={latency_ms}ms tps={tps:.1f}"
        )

        bus = get_event_bus()
        if bus is not None:
            bus.publish({
                "type": "billing",
                "request_id": request_id,
                "consumer_id": consumer_id,
                "provider_id": provider.provider_id,
                "provider_name": provider.name,
                "model": model,
                "cost_usd": round(
                    (output_tokens * provider.capabilities.price_per_mtok_output) / 1_000_000, 6
                ),
                "tokens": output_tokens,
                "input_tokens": actual_input,
                "cached_tokens": actual_cached,
                "latency_ms": latency_ms,
                "tps": round(tps, 2),
                "trust_level": provider.capabilities.trust_level.value,
                "encrypted": bool(provider.encryption_public_key),
            })

        if output_tokens > 0 and elapsed > 0:
            tps_tracker = get_tps_tracker()
            tps_tracker.record_request(
                provider_id=provider.provider_id,
                model=model,
                tokens=output_tokens,
                seconds=elapsed,
                hardware=provider.capabilities.hardware,
            )

    return callback


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    hub = get_hub()
    auth = get_auth()

    # Input validation
    validation_error = request.validate_inputs()
    if validation_error:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": {"message": validation_error, "type": "invalid_request"}}, status_code=400)

    # Resolve consumer identity from auth header or JWT
    from .routes_auth import (
        anonymous_inference_blocked,
        resolve_user_from_request,
        sign_in_required_response,
    )
    user_info = resolve_user_from_request(raw_request)
    if user_info:
        consumer_id = user_info["user_id"]
    else:
        consumer_id = auth.resolve_consumer(raw_request.headers.get("authorization"))

    if anonymous_inference_blocked(consumer_id):
        return sign_in_required_response()

    # Rate limit check
    if not _rate_limiter.allow(consumer_id):
        raise RateLimitExceeded()

    # Balance check: reject if consumer has no credits
    from .dependencies import get_store
    store = get_store()
    account = store.get_account(consumer_id)
    if account and account["balance_micro"] <= 0:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"error": {"message": "Insufficient balance. Add credits to continue.", "type": "insufficient_balance"}},
            status_code=402,
        )

    # Pre-compute input token estimate for billing (used by both paths)
    messages_plain = [m.model_dump() for m in request.messages]
    input_token_estimate = _estimate_input_tokens(messages_plain)

    # Select a provider
    provider = hub.select_provider(
        request.model,
        preference=request.ocip_preference,
        min_confidence=request.ocip_min_confidence,
        max_price=request.ocip_max_price,
        session_id=request.ocip_session_id,
        reputation_fn=get_reputation_tracker().get_score,
    )
    if provider is None:
        # No provider available right now — try queuing
        request_id = str(uuid.uuid4())

        # Build the inference request early so it's ready when dispatched
        inference_req = InferenceRequest(
            request_id=request_id,
            model=request.model,
            messages=messages_plain,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream,
        )

        try:
            pending = hub.enqueue_request(
                inference_req,
                model=request.model,
                preference=request.ocip_preference,
                min_confidence=request.ocip_min_confidence,
                max_price=request.ocip_max_price,
                session_id=request.ocip_session_id,
            )
        except asyncio.QueueFull:
            _add_trace({
                "request_id": request_id[:8],
                "timestamp": time.time(),
                "model": request.model,
                "status": "queue_full",
                "reason": f"Queue full ({hub.QUEUE_MAX_DEPTH} pending)",
                "providers_evaluated": hub.provider_count,
            }, consumer_id)
            raise QueueFull(hub.QUEUE_MAX_DEPTH)

        # Wait for a provider to be assigned (up to timeout)
        try:
            await asyncio.wait_for(
                pending.event.wait(),
                timeout=hub.QUEUE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            _add_trace({
                "request_id": request_id[:8],
                "timestamp": time.time(),
                "model": request.model,
                "status": "queue_timeout",
                "reason": f"No provider available after {hub.QUEUE_TIMEOUT_SECONDS}s wait",
                "providers_evaluated": hub.provider_count,
            }, consumer_id)
            raise QueueTimeout(hub.QUEUE_TIMEOUT_SECONDS)

        # Provider was assigned by the dispatch loop
        provider = pending.provider
        if provider is None:
            raise NoProviderAvailable("Provider assignment failed.")

        logger.info(
            f"[{request_id[:8]}] Dequeued → {provider.name} "
            f"(waited {time.time() - pending.queued_at:.1f}s)"
        )

        # Re-encrypt if needed (the initial inference_req was built with plaintext)
        if provider.encryption_public_key:
            encrypt_payload = {"messages": messages_plain}
            if request.ocip_consumer_public_key:
                encrypt_payload["consumer_public_key"] = request.ocip_consumer_public_key
            encrypted_body = encrypt_json(
                encrypt_payload,
                provider.encryption_public_key,
            ).to_dict()
            inference_req = InferenceRequest(
                request_id=request_id,
                model=request.model,
                messages=None,
                encrypted_body=encrypted_body,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream,
            )
            logger.info(f"[{request_id[:8]}] 🔐 Request encrypted to provider")

        # Now send to the assigned provider
        queue = hub.create_response_queue(request_id)
        try:
            await hub.send_to_provider(provider, inference_req)
        except Exception as e:
            hub.remove_response_queue(request_id)
            raise ProviderError(f"Failed to reach provider: {e}")

        # Build scoring trace
        scoring_details = []
        for p in hub._providers.values():
            score = p.score_for_request(request.model, request.ocip_preference)
            scoring_details.append({
                "provider_id": p.provider_id,
                "name": p.name,
                "price": p.capabilities.price_per_mtok_output,
                "trust": p.capabilities.trust_level.value,
                "load": round(p.load_factor, 2),
                "tps": p.capabilities.measured_tps,
                "score": round(score, 4),
                "selected": p.provider_id == provider.provider_id,
                "encrypted": bool(p.encryption_public_key),
            })
        scoring_details.sort(key=lambda x: x["score"], reverse=True)

        _add_trace({
            "request_id": request_id[:8],
            "timestamp": time.time(),
            "model": request.model,
            "preference": request.ocip_preference,
            "min_confidence": request.ocip_min_confidence,
            "max_price": request.ocip_max_price,
            "status": "matched_from_queue",
            "wait_seconds": round(time.time() - pending.queued_at, 2),
            "selected_provider": provider.name,
            "selected_price": provider.capabilities.price_per_mtok_output,
            "selected_price_input": provider.capabilities.price_per_mtok_input,
            "selected_price_cache": provider.capabilities.price_per_mtok_cache,
            "selected_trust": provider.capabilities.trust_level.value,
            "encrypted": bool(provider.encryption_public_key),
            "scoring": scoring_details,
            "providers_evaluated": len(scoring_details),
        }, consumer_id)

        # Publish match event
        bus = get_event_bus()
        if bus is not None:
            bus.publish({
                "type": "match",
                "request_id": request_id,
                "provider": provider.name,
                "model": request.model,
                "source": "queue",
            })

        # Register billing callback so InferenceDone triggers billing
        # even if the streaming generator is cancelled by FastAPI
        start_time = time.time()
        queue_wait = int((time.time() - pending.queued_at) * 1000)
        hub.register_billing_callback(
            request_id,
            _make_billing_callback(
                request_id, request.model, provider, consumer_id, input_token_estimate, start_time,
                preference=request.ocip_preference, queued=True, queue_wait_ms=queue_wait,
            ),
        )

        if request.stream:
            return StreamingResponse(
                _stream_response(request_id, request.model, queue, hub, provider, consumer_id, start_time),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-OCIP-Provider": provider.name,
                    "X-OCIP-Trust-Level": provider.capabilities.trust_level.value,
                    "X-OCIP-Price-Output": str(provider.capabilities.price_per_mtok_output),
                    "X-OCIP-Queued": "true",
                },
            )
        else:
            return await _collect_response(request_id, request.model, queue, hub, provider, consumer_id, start_time)

    # Build scoring trace for this decision
    scoring_details = []
    for p in hub._providers.values():
        score = p.score_for_request(request.model, request.ocip_preference)
        scoring_details.append({
            "provider_id": p.provider_id,
            "name": p.name,
            "price": p.capabilities.price_per_mtok_output,
            "trust": p.capabilities.trust_level.value,
            "load": round(p.load_factor, 2),
            "tps": p.capabilities.measured_tps,
            "score": round(score, 4),
            "selected": p.provider_id == provider.provider_id,
            "encrypted": bool(p.encryption_public_key),
        })
    scoring_details.sort(key=lambda x: x["score"], reverse=True)

    request_id = str(uuid.uuid4())

    # Create response queue before sending request
    queue = hub.create_response_queue(request_id)

    # Build request — encrypt if provider supports E2E
    encrypted_body = None

    if provider.encryption_public_key:
        # OCIP E2E: encrypt messages to provider's key
        # Include consumer's public key so provider can encrypt responses back
        encrypt_payload = {"messages": messages_plain}
        if request.ocip_consumer_public_key:
            encrypt_payload["consumer_public_key"] = request.ocip_consumer_public_key
        encrypted_body = encrypt_json(
            encrypt_payload,
            provider.encryption_public_key,
        ).to_dict()
        logger.info(f"[{request_id[:8]}] 🔐 Request encrypted to provider")

    inference_req = InferenceRequest(
        request_id=request_id,
        model=request.model,
        messages=messages_plain if not encrypted_body else None,
        encrypted_body=encrypted_body,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream,
    )

    try:
        await hub.send_to_provider(provider, inference_req)
    except Exception as e:
        hub.remove_response_queue(request_id)
        # RETRY: try another provider
        logger.warning(f"[{request_id[:8]}] Provider {provider.name} unreachable, retrying...")
        retry_provider = hub.select_provider(
            request.model,
            preference=request.ocip_preference,
            min_confidence=request.ocip_min_confidence,
            max_price=request.ocip_max_price,
        )
        if retry_provider and retry_provider.provider_id != provider.provider_id:
            queue = hub.create_response_queue(request_id)
            provider = retry_provider  # Use the retry provider for billing/traces
            try:
                await hub.send_to_provider(retry_provider, inference_req)
                logger.info(f"[{request_id[:8]}] Retried on {retry_provider.name}")
            except Exception as e2:
                hub.remove_response_queue(request_id)
                raise ProviderError(f"All providers failed: {e2}")
        else:
            raise ProviderError(f"Failed to reach provider: {e}")

    # Log the full decision trace
    _add_trace({
        "request_id": request_id[:8],
        "timestamp": time.time(),
        "model": request.model,
        "preference": request.ocip_preference,
        "min_confidence": request.ocip_min_confidence,
        "max_price": request.ocip_max_price,
        "status": "matched",
        "selected_provider": provider.name,
        "selected_price": provider.capabilities.price_per_mtok_output,
        "selected_price_input": provider.capabilities.price_per_mtok_input,
        "selected_price_cache": provider.capabilities.price_per_mtok_cache,
        "selected_trust": provider.capabilities.trust_level.value,
        "encrypted": bool(provider.encryption_public_key),
        "scoring": scoring_details,
        "providers_evaluated": len(scoring_details),
    }, consumer_id)

    # Publish match event
    bus = get_event_bus()
    if bus is not None:
        bus.publish({
            "type": "match",
            "request_id": request_id,
            "provider": provider.name,
            "model": request.model,
            "score": max((s["score"] for s in scoring_details if s["selected"]), default=0),
        })

    # Register billing callback so InferenceDone triggers billing
    # even if the streaming generator is cancelled by FastAPI
    start_time = time.time()
    hub.register_billing_callback(
        request_id,
        _make_billing_callback(
            request_id, request.model, provider, consumer_id, input_token_estimate, start_time,
            preference=request.ocip_preference,
        ),
    )

    if request.stream:
        return StreamingResponse(
            _stream_response(request_id, request.model, queue, hub, provider, consumer_id, start_time),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-OCIP-Provider": provider.name,
                "X-OCIP-Trust-Level": provider.capabilities.trust_level.value,
                "X-OCIP-Price-Output": str(provider.capabilities.price_per_mtok_output),
            },
        )
    else:
        # Non-streaming: collect all tokens and return as one response
        return await _collect_response(request_id, request.model, queue, hub, provider, consumer_id, start_time)


async def _stream_response(
    request_id: str, model: str, queue: asyncio.Queue, hub, provider, consumer_id: str,
    start_time: float,
):
    """Generate SSE stream from provider response chunks.

    Billing is handled by the billing callback registered in the hub,
    which fires when InferenceDone arrives via WebSocket — not here.
    This avoids the race where FastAPI cancels this generator before
    InferenceDone is read from the queue.
    """
    token_count = 0
    outcome = "success"
    completed = False
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                outcome = "timeout"
                error_data = {
                    "error": {"message": "Provider timeout", "type": "timeout"}
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                break

            if isinstance(msg, InferenceResponseChunk):
                token_count += 1
                chunk = {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": msg.token} if msg.token else {},
                            "finish_reason": msg.finish_reason,
                        }
                    ],
                }
                if msg.encrypted_token:
                    chunk["ocip_encrypted_token"] = msg.encrypted_token
                yield f"data: {json.dumps(chunk)}\n\n"

                # Don't break on finish_reason — wait for InferenceDone

            elif isinstance(msg, InferenceDone):
                completed = True
                chunk = {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                break

            elif isinstance(msg, InferenceError):
                completed = True
                outcome = "disconnect" if msg.error == "provider_disconnected" else "error"
                error_data = {"error": {"message": msg.error, "type": "provider_error"}}
                yield f"data: {json.dumps(error_data)}\n\n"
                break

        yield "data: [DONE]\n\n"

        # Record reputation outcome
        reputation = get_reputation_tracker()
        elapsed_ms = int((time.time() - start_time) * 1000)
        if outcome == "success":
            reputation.record_success(provider.provider_id, tokens=token_count, latency_ms=elapsed_ms)
        elif outcome == "timeout":
            reputation.record_timeout(provider.provider_id)
        elif outcome == "disconnect":
            reputation.record_disconnect(provider.provider_id)
        elif outcome == "error":
            reputation.record_error(provider.provider_id)
    finally:
        if not completed:
            try:
                await hub.cancel_request(request_id, output_tokens=token_count)
            except Exception as e:
                logger.warning(f"[{request_id[:8]}] Failed to cancel provider request: {e}")
        hub.remove_response_queue(request_id)


async def _collect_response(
    request_id: str, model: str, queue: asyncio.Queue, hub, provider, consumer_id: str,
    start_time: float,
) -> dict:
    """Collect all tokens into a single non-streaming response.

    Billing is handled by the billing callback registered in the hub.
    """
    tokens: list[str] = []
    done_msg_ns: InferenceDone | None = None
    completed = False
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                reputation = get_reputation_tracker()
                reputation.record_timeout(provider.provider_id)
                raise ProviderTimeout()

            if isinstance(msg, InferenceResponseChunk):
                tokens.append(msg.token)
                # Don't break on finish_reason — wait for InferenceDone
            elif isinstance(msg, InferenceDone):
                done_msg_ns = msg
                completed = True
                break
            elif isinstance(msg, InferenceError):
                completed = True
                reputation = get_reputation_tracker()
                reputation.record_error(provider.provider_id)
                raise ProviderError(msg.error)
    finally:
        if not completed:
            try:
                await hub.cancel_request(request_id, output_tokens=len(tokens))
            except Exception as e:
                logger.warning(f"[{request_id[:8]}] Failed to cancel provider request: {e}")
        hub.remove_response_queue(request_id)

    # Record reputation — success
    reputation = get_reputation_tracker()
    elapsed_ms = int((time.time() - start_time) * 1000)
    reputation.record_success(provider.provider_id, tokens=len(tokens), latency_ms=elapsed_ms)

    content = "".join(tokens)
    return {
        "id": f"chatcmpl-{request_id[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": done_msg_ns.input_tokens if done_msg_ns and done_msg_ns.input_tokens > 0 else len(content) // 4,
            "completion_tokens": len(tokens),
            "total_tokens": (done_msg_ns.input_tokens if done_msg_ns and done_msg_ns.input_tokens > 0 else len(content) // 4) + len(tokens),
        },
    }
