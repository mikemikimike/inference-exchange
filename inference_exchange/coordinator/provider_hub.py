"""Manages WebSocket connections to providers and routes requests."""

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import WebSocket

from inference_exchange.shared.protocol import (
    AttestationChallenge,
    AttestationResponse,
    CancelRequest,
    HeartbeatMessage,
    InferenceDone,
    InferenceError,
    InferenceRequest,
    InferenceResponseChunk,
    MessageType,
    ModelCapabilities,
    ProviderCapabilities,
    RegisterMessage,
)

logger = logging.getLogger(__name__)


@dataclass
class ConnectedProvider:
    provider_id: str
    name: str
    ws: WebSocket
    capabilities: ProviderCapabilities
    encryption_public_key: str = ""
    connected_at: float = field(default_factory=time.time)
    active_requests: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    model_verified: bool = False
    model_identity: dict = field(default_factory=dict)
    model_capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    attestation_status: str = "pending"
    app_attest_verified: bool = False
    admission_expires_at: float = 0.0
    app_attest_key_id: str = ""
    provider_artifact_hash: str = ""
    last_attestation: float = 0.0
    pending_challenge_nonce: str | None = None
    challenge_sent_at: float = 0.0

    @property
    def requires_app_attest(self) -> bool:
        return self.capabilities.trust_level.value == "confidential"

    @property
    def admitted(self) -> bool:
        if self.requires_app_attest:
            return self.app_attest_verified and self.admission_expires_at > time.time()
        return True

    @property
    def verified_trust_level(self) -> str | None:
        """Trust level backed by evidence, or None. Single hook point for L2/L3 verification (#1)."""
        if self.requires_app_attest and self.admitted:
            return self.capabilities.trust_level.value
        return None

    @property
    def load_factor(self) -> float:
        if self.capabilities.max_concurrent == 0:
            return 1.0
        return self.active_requests / self.capabilities.max_concurrent

    def score_for_request(self, model: str, preference: str = "balanced") -> float:
        from .matching.models import (
            ConfidenceLevel,
            InferenceOrder,
            ProviderOffer,
            RoutingPreference,
        )
        from .matching.strategy import compute_score

        pref_map = {
            "cheapest": RoutingPreference.CHEAPEST,
            "fastest": RoutingPreference.FASTEST,
            "most_secure": RoutingPreference.MOST_SECURE,
            "balanced": RoutingPreference.BALANCED,
        }
        conf_map = {"open": ConfidenceLevel.OPEN, "contained": ConfidenceLevel.CONTAINED,
                     "hardened": ConfidenceLevel.HARDENED, "confidential": ConfidenceLevel.CONFIDENTIAL}
        order = InferenceOrder(order_id="trace", consumer_id="", model=model,
                               preference=pref_map.get(preference, RoutingPreference.BALANCED))
        offer = ProviderOffer(
            provider_id=self.provider_id,
            provider_name=self.name,
            models=self.capabilities.models,
            price_per_mtok_input=self.capabilities.price_per_mtok_input,
            price_per_mtok_output=self.capabilities.price_per_mtok_output,
            confidence_level=conf_map.get(self.capabilities.trust_level.value, ConfidenceLevel.OPEN),
            measured_throughput_tps=self.capabilities.measured_tps,
            total_slots=self.capabilities.max_concurrent,
            used_slots=self.active_requests,
            encrypted=bool(self.encryption_public_key),
            hardware=self.capabilities.hardware,
        )
        return compute_score(order, offer)


@dataclass
class PendingRequest:
    request_id: str
    inference_req: InferenceRequest
    event: asyncio.Event
    provider: ConnectedProvider | None = None
    queued_at: float = field(default_factory=time.time)
    model: str = "default"
    preference: str = "balanced"
    min_confidence: str = "open"
    max_price: float | None = None
    session_id: str | None = None


class ProviderHub:
    QUEUE_MAX_DEPTH = 50
    QUEUE_TIMEOUT_SECONDS = 30.0

    def __init__(self):
        self._providers: dict[str, ConnectedProvider] = {}
        self._response_queues: dict[str, asyncio.Queue] = {}
        self._next_provider_id = 0
        self._session_affinity: dict[str, str] = {}
        self._SESSION_AFFINITY_MAX = 1000
        self._request_to_provider: dict[str, str] = {}
        self._pending_queue: asyncio.Queue[PendingRequest] = asyncio.Queue(maxsize=self.QUEUE_MAX_DEPTH)
        # Billing callbacks keyed by request_id, invoked when InferenceDone arrives
        self._billing_callbacks: dict[str, Callable[[InferenceDone], None]] = {}
        self._delivered_tokens: dict[str, int] = {}
        self._terminal_requests: set[str] = set()

    @property
    def provider_count(self) -> int:
        return len(self._providers)

    @property
    def available_models(self) -> list[str]:
        models = set()
        for p in self._providers.values():
            models.update(p.capabilities.models)
        return sorted(models)

    def register_provider(self, ws: WebSocket, msg: RegisterMessage) -> str:
        self._next_provider_id += 1
        provider_id = f"provider-{self._next_provider_id}"
        self._providers[provider_id] = ConnectedProvider(
            provider_id=provider_id,
            name=msg.provider_name,
            ws=ws,
            capabilities=msg.capabilities,
            encryption_public_key=msg.encryption_public_key,
            app_attest_key_id=msg.app_attest_key_id,
            provider_artifact_hash=msg.provider_artifact_hash,
        )
        encrypted_status = "🔐 E2E" if msg.encryption_public_key else "🔓 plaintext"
        logger.info(
            f"Provider registered: {msg.provider_name} ({provider_id}) "
            f"models={msg.capabilities.models} trust={msg.capabilities.trust_level} [{encrypted_status}]"
        )
        return provider_id

    def disconnect_provider(self, provider_id: str):
        if provider_id in self._providers:
            provider = self._providers[provider_id]
            name = provider.name
            orphaned_requests = [req_id for req_id, pid in self._request_to_provider.items() if pid == provider_id]
            for req_id in orphaned_requests:
                self._terminal_requests.add(req_id)
                if req_id in self._response_queues:
                    self._response_queues[req_id].put_nowait(
                        InferenceError(request_id=req_id, error="provider_disconnected")
                    )
                callback = self._billing_callbacks.pop(req_id, None)
                if callback:
                    try:
                        callback(InferenceDone(
                            request_id=req_id,
                            tokens_generated=self._delivered_tokens.get(req_id, 0),
                        ))
                    except Exception as e:
                        logger.error(f"Billing callback error for {req_id[:8]}: {e}")
                self._request_to_provider.pop(req_id, None)
            del self._providers[provider_id]
            logger.info(f"Provider disconnected: {name} ({provider_id})")
            return orphaned_requests
        return []

    def select_provider(self, model: str, preference: str = "balanced",
                        min_confidence: str = "open", max_price: float | None = None,
                        session_id: str | None = None, reputation_fn=None) -> ConnectedProvider | None:
        from .matching.models import (
            ConfidenceLevel,
            InferenceOrder,
            ProviderOffer,
            RoutingPreference,
        )
        from .matching.strategy import GreedyStrategy

        pref_map = {"cheapest": RoutingPreference.CHEAPEST, "fastest": RoutingPreference.FASTEST,
                    "most_secure": RoutingPreference.MOST_SECURE, "balanced": RoutingPreference.BALANCED}
        conf_map = {"open": ConfidenceLevel.OPEN, "contained": ConfidenceLevel.CONTAINED,
                     "hardened": ConfidenceLevel.HARDENED, "confidential": ConfidenceLevel.CONFIDENTIAL}
        affinity_pid = self._session_affinity.get(session_id, "") if session_id else ""
        order = InferenceOrder(
            order_id="select", consumer_id="", model=model,
            max_price_per_mtok=max_price if max_price is not None else float("inf"),
            min_confidence=conf_map.get(min_confidence, ConfidenceLevel.OPEN),
            preference=pref_map.get(preference, RoutingPreference.BALANCED),
            session_affinity_provider_id=affinity_pid,
        )

        offers = []
        for p in self._providers.values():
            # Confidential providers are invisible to routing until App Attest admission succeeds.
            if not p.admitted:
                continue
            rep = reputation_fn(p.provider_id) if reputation_fn else 1.0
            offers.append(ProviderOffer(
                provider_id=p.provider_id,
                provider_name=p.name,
                models=p.capabilities.models,
                price_per_mtok_input=p.capabilities.price_per_mtok_input,
                price_per_mtok_output=p.capabilities.price_per_mtok_output,
                confidence_level=conf_map.get(p.capabilities.trust_level.value, ConfidenceLevel.OPEN),
                measured_throughput_tps=p.capabilities.measured_tps,
                total_slots=p.capabilities.max_concurrent,
                used_slots=p.active_requests,
                encrypted=bool(p.encryption_public_key),
                hardware=p.capabilities.hardware,
                reputation_score=rep,
            ))
        matches, _ = GreedyStrategy().match([order], offers)
        if matches:
            winner_id = matches[0].provider_id
            if winner_id in self._providers:
                if session_id:
                    self._session_affinity[session_id] = winner_id
                    if len(self._session_affinity) > self._SESSION_AFFINITY_MAX:
                        del self._session_affinity[next(iter(self._session_affinity))]
                return self._providers[winner_id]
        return None

    def create_response_queue(self, request_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._response_queues[request_id] = queue
        self._delivered_tokens[request_id] = 0
        self._terminal_requests.discard(request_id)
        return queue

    def remove_response_queue(self, request_id: str):
        self._response_queues.pop(request_id, None)
        self._request_to_provider.pop(request_id, None)
        self._billing_callbacks.pop(request_id, None)
        self._delivered_tokens.pop(request_id, None)
        self._terminal_requests.discard(request_id)

    def register_billing_callback(
        self, request_id: str, callback: Callable[[InferenceDone], None]
    ):
        """Register a callback invoked when InferenceDone arrives for this request."""
        self._billing_callbacks[request_id] = callback

    async def cancel_request(self, request_id: str, output_tokens: int | None = None) -> bool:
        """Stop an active provider request and settle billing for delivered output."""
        if request_id in self._terminal_requests:
            return False

        delivered = self._delivered_tokens.get(request_id, 0)
        output_tokens = delivered if output_tokens is None else max(0, output_tokens)
        callback = self._billing_callbacks.pop(request_id, None)
        if callback:
            try:
                callback(InferenceDone(request_id=request_id, tokens_generated=output_tokens))
            except Exception as e:
                logger.error(f"Billing callback error for {request_id[:8]}: {e}")

        provider_id = self._request_to_provider.get(request_id)
        provider = self._providers.get(provider_id) if provider_id else None
        if provider is None:
            return callback is not None

        try:
            await provider.ws.send_json(CancelRequest(request_id=request_id).model_dump())
        except Exception as e:
            logger.warning(f"Cancel request failed for {request_id[:8]}: {e}")
        provider.active_requests = max(0, provider.active_requests - 1)
        asyncio.create_task(self._try_dispatch_queued(freed_provider_id=provider.provider_id))
        return True

    async def send_to_provider(self, provider: ConnectedProvider, request: InferenceRequest):
        if not provider.admitted and provider.requires_app_attest:
            raise PermissionError("provider is not App Attest admitted")
        provider.active_requests += 1
        self._request_to_provider[request.request_id] = provider.provider_id
        try:
            await provider.ws.send_json(request.model_dump())
        except Exception:
            provider.active_requests = max(0, provider.active_requests - 1)
            self._request_to_provider.pop(request.request_id, None)
            raise

    def enqueue_request(self, inference_req: InferenceRequest, *, model: str = "default",
                        preference: str = "balanced", min_confidence: str = "open",
                        max_price: float | None = None, session_id: str | None = None) -> PendingRequest:
        pending = PendingRequest(request_id=inference_req.request_id, inference_req=inference_req,
                                 event=asyncio.Event(), model=model, preference=preference,
                                 min_confidence=min_confidence, max_price=max_price, session_id=session_id)
        self._pending_queue.put_nowait(pending)
        return pending

    @property
    def pending_queue_size(self) -> int:
        return self._pending_queue.qsize()

    async def _try_dispatch_queued(self, freed_provider_id: str | None = None):
        if self._pending_queue.empty():
            return
        items: list[PendingRequest] = []
        dispatched = False
        while not self._pending_queue.empty():
            try:
                items.append(self._pending_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for pending in items:
            if dispatched:
                self._pending_queue.put_nowait(pending)
                continue
            provider = self.select_provider(pending.model, preference=pending.preference,
                                            min_confidence=pending.min_confidence,
                                            max_price=pending.max_price, session_id=pending.session_id)
            if provider is not None:
                pending.provider = provider
                pending.event.set()
                dispatched = True
            else:
                self._pending_queue.put_nowait(pending)

    def handle_provider_message(self, provider_id: str, data: dict):
        msg_type = data.get("type")
        request_id = data.get("request_id")
        if not request_id:
            return

        # InferenceDone billing must fire even after the response queue is
        # cleaned up (the streaming generator may have finished and removed
        # the queue before InferenceDone arrives from the provider).
        if msg_type == MessageType.INFERENCE_DONE:
            done_msg = InferenceDone(**data)
            self._terminal_requests.add(request_id)
            # Deliver to queue if it still exists (streaming generator reads it)
            queue = self._response_queues.get(request_id)
            if queue is not None:
                queue.put_nowait(done_msg)
            # Fire billing callback regardless of queue state
            callback = self._billing_callbacks.pop(request_id, None)
            if callback:
                try:
                    callback(done_msg)
                except Exception as e:
                    logger.error(f"Billing callback error for {request_id[:8]}: {e}")
            if provider_id in self._providers:
                self._providers[provider_id].active_requests = max(0, self._providers[provider_id].active_requests - 1)
            asyncio.ensure_future(self._try_dispatch_queued(freed_provider_id=provider_id))
            return

        # All other message types require an active response queue
        if request_id not in self._response_queues:
            return
        queue = self._response_queues[request_id]
        if msg_type == MessageType.INFERENCE_RESPONSE:
            queue.put_nowait(InferenceResponseChunk(**data))
            self._delivered_tokens[request_id] = self._delivered_tokens.get(request_id, 0) + 1
        elif msg_type == MessageType.INFERENCE_ERROR:
            queue.put_nowait(InferenceError(**data))
            self._terminal_requests.add(request_id)
            callback = self._billing_callbacks.pop(request_id, None)
            if callback:
                try:
                    callback(InferenceDone(
                        request_id=request_id,
                        tokens_generated=self._delivered_tokens.get(request_id, 0),
                    ))
                except Exception as e:
                    logger.error(f"Billing callback error for {request_id[:8]}: {e}")
            if provider_id in self._providers:
                self._providers[provider_id].active_requests = max(0, self._providers[provider_id].active_requests - 1)
            asyncio.ensure_future(self._try_dispatch_queued(freed_provider_id=provider_id))

    def handle_heartbeat(self, provider_id: str, msg: HeartbeatMessage):
        if provider_id in self._providers:
            provider = self._providers[provider_id]
            provider.active_requests = msg.active_requests
            provider.last_heartbeat = time.time()

    async def send_attestation_challenge(self, provider: ConnectedProvider) -> str:
        nonce = secrets.token_hex(32)
        challenge = AttestationChallenge(nonce=nonce, timestamp=time.time(), required_profile=provider.capabilities.trust_level.value)
        provider.pending_challenge_nonce = nonce
        provider.challenge_sent_at = time.time()
        await provider.ws.send_json(challenge.model_dump())
        return nonce

    def handle_attestation_response(self, provider_id: str, data: dict):
        if provider_id not in self._providers:
            return
        provider = self._providers[provider_id]
        response = AttestationResponse(**data)
        if not provider.pending_challenge_nonce or response.nonce != provider.pending_challenge_nonce:
            provider.attestation_status = "degraded"
            provider.app_attest_verified = False
            return
        provider.pending_challenge_nonce = None
        provider.last_attestation = time.time()
        trust_claim = provider.capabilities.trust_level.value
        issues = []
        if trust_claim in ("hardened", "confidential"):
            if not response.sip_enabled:
                issues.append("SIP disabled")
            if not response.hardened_runtime:
                issues.append("no Hardened Runtime")
            if not response.agent_binary_hash:
                issues.append("agent not frozen")
        if trust_claim == "confidential" and not response.app_attest_attestation_object:
            issues.append("missing cryptographic App Attest attestation")
        if issues:
            provider.attestation_status = "degraded"
            provider.app_attest_verified = False
        else:
            # Cryptographic App Attest verification is performed by the coordinator admission service.
            # Do not turn self-reported `hardware_attestation` into an admission grant.
            provider.attestation_status = "passed" if trust_claim != "confidential" else "pending"
            provider.app_attest_verified = False

    def check_attestation_timeouts(self, timeout_seconds: float = 30.0):
        now = time.time()
        for provider in self._providers.values():
            if provider.pending_challenge_nonce and now - provider.challenge_sent_at > timeout_seconds:
                provider.pending_challenge_nonce = None
                provider.attestation_status = "degraded"
                provider.app_attest_verified = False
