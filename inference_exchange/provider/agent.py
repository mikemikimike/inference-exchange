"""DEPRECATED: Use ocip_agent/agent.py instead.

This is the simple single-process provider agent that runs inference
in-process via llama-cpp-python. It does NOT support:
- Two-process architecture (hardened server isolation)
- Response encryption (consumer keypair)
- GGUF metadata-based model identity
- Attestation with hardening evidence
- Provider token authentication

Use this only for quick local testing. For production, use:
  python ocip_agent/agent.py --name my-node --coordinator ws://...

Kept for backward compatibility and the docker-compose demo.
"""

import asyncio
import json
import logging
import time

import websockets

from inference_exchange.config import ProviderConfig
from inference_exchange.shared.crypto import EncryptedPayload, KeyPair, decrypt_json
from inference_exchange.shared.protocol import (
    HeartbeatMessage,
    InferenceDone,
    InferenceError,
    InferenceRequest,
    InferenceResponseChunk,
    MessageType,
    ProviderCapabilities,
    RegisterMessage,
    TrustLevel,
)

logger = logging.getLogger(__name__)


class ProviderAgent:
    """Connects to coordinator and serves inference requests."""

    def __init__(
        self,
        config: ProviderConfig,
        engine,  # InferenceEngine or MockEngine
        price_per_mtok_input: float = 0.05,
        price_per_mtok_output: float = 0.20,
        price_per_mtok_cache: float = 0,
        trust_level: str = "open",
        measured_tps: float = 0,
        hardware_override: str | None = None,
        model_names_override: list[str] | None = None,
        model_identity: dict | None = None,
    ):
        self.config = config
        self.engine = engine
        self.price_per_mtok_input = price_per_mtok_input
        self.price_per_mtok_output = price_per_mtok_output
        self.price_per_mtok_cache = price_per_mtok_cache
        self.trust_level = trust_level
        self.measured_tps = measured_tps
        self.hardware_override = hardware_override
        self.model_names_override = model_names_override
        self.model_identity = model_identity or {}
        self._ws = None
        self._active_requests = 0
        self._active_request_tasks: dict[str, asyncio.Task] = {}
        self._running = False
        # Generate X25519 keypair for E2E encryption
        self._keypair = KeyPair()
        logger.info(f"🔐 Encryption key generated: {self._keypair.public_key_b64[:16]}...")

    async def run(self):
        """Main loop: connect to coordinator and process requests."""
        self._running = True

        while self._running:
            try:
                await self._connect_and_serve()
            except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
                logger.warning(f"Connection lost: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)

    async def _connect_and_serve(self):
        """Connect to coordinator and enter the message loop."""
        logger.info(f"Connecting to coordinator: {self.config.coordinator_url}")

        async with websockets.connect(self.config.coordinator_url) as ws:
            self._ws = ws
            logger.info("Connected to coordinator")

            # Register
            model_list = self.model_names_override or [self.engine.model_name]
            reg = RegisterMessage(
                provider_name=self.config.provider_name,
                capabilities=ProviderCapabilities(
                    models=model_list,
                    max_concurrent=self.config.max_concurrent,
                    trust_level=TrustLevel(self.trust_level),
                    hardware=self.hardware_override or self._detect_hardware(),
                    measured_tps=self.measured_tps,
                    price_per_mtok_input=self.price_per_mtok_input,
                    price_per_mtok_output=self.price_per_mtok_output,
                    price_per_mtok_cache=self.price_per_mtok_cache,
                ),
                encryption_public_key=self._keypair.public_key_b64,
            )
            await ws.send(reg.model_dump_json())
            logger.info(f"Registered as: {self.config.provider_name}")

            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))

            try:
                # Message loop
                async for raw in ws:
                    data = json.loads(raw)
                    msg_type = data.get("type")

                    if msg_type == MessageType.INFERENCE_REQUEST:
                        req = InferenceRequest(**data)
                        # Handle inference in a separate task so we can process
                        # multiple requests concurrently
                        task = asyncio.create_task(self._handle_inference(ws, req))
                        self._active_request_tasks[req.request_id] = task

                    elif msg_type == MessageType.CANCEL_REQUEST:
                        request_id = data.get("request_id", "")
                        task = self._active_request_tasks.get(request_id)
                        if task:
                            task.cancel()
                            logger.info(f"[{request_id[:8]}] Cancelled")

                    else:
                        logger.debug(f"Unknown message type: {msg_type}")
            finally:
                heartbeat_task.cancel()
                for task in self._active_request_tasks.values():
                    task.cancel()
                self._active_request_tasks.clear()

    async def _handle_inference(self, ws, req: InferenceRequest):
        """Run inference and stream results back to coordinator."""
        self._active_requests += 1
        start = time.time()
        tokens_generated = 0

        logger.info(f"[{req.request_id[:8]}] Inference started: {req.model}")

        try:
            # Decrypt messages if E2E encrypted
            if req.encrypted_body:
                payload = EncryptedPayload.from_dict(req.encrypted_body)
                decrypted = decrypt_json(payload, self._keypair.private_key)
                messages = decrypted["messages"]
                logger.info(f"[{req.request_id[:8]}] 🔐 Decrypted E2E payload")
            elif req.messages:
                messages = req.messages
            else:
                raise ValueError("No messages or encrypted_body in request")

            # Run inference in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()

            # Stream tokens
            for token in await loop.run_in_executor(
                None, lambda: list(self.engine.generate_stream(
                    messages=messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                ))
            ):
                chunk = InferenceResponseChunk(
                    request_id=req.request_id,
                    token=token,
                )
                await ws.send(chunk.model_dump_json())
                tokens_generated += 1

            # Send done
            elapsed = time.time() - start
            done = InferenceDone(
                request_id=req.request_id,
                tokens_generated=tokens_generated,
                time_seconds=elapsed,
            )
            await ws.send(done.model_dump_json())

            tps = tokens_generated / elapsed if elapsed > 0 else 0
            logger.info(
                f"[{req.request_id[:8]}] Done: {tokens_generated} tokens "
                f"in {elapsed:.1f}s ({tps:.1f} tok/s)"
            )

        except asyncio.CancelledError:
            logger.info(f"[{req.request_id[:8]}] Cancelled by coordinator")
            try:
                await ws.send(InferenceError(
                    request_id=req.request_id,
                    error="cancelled",
                ).model_dump_json())
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"[{req.request_id[:8]}] Inference error: {e}")
            error = InferenceError(request_id=req.request_id, error=str(e))
            try:
                await ws.send(error.model_dump_json())
            except Exception:
                pass
        finally:
            self._active_requests -= 1
            self._active_request_tasks.pop(req.request_id, None)

    async def _heartbeat_loop(self, ws):
        """Send periodic heartbeats to coordinator."""
        while True:
            try:
                hb = HeartbeatMessage(
                    active_requests=self._active_requests,
                    loaded_models=[self.engine.model_name],
                )
                await ws.send(hb.model_dump_json())
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception:
                break

    def _detect_hardware(self) -> str:
        """Detect hardware platform."""
        import platform

        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "darwin":
            # Could detect Apple Silicon specifics here
            return f"apple-silicon-{machine}"
        elif "amd" in platform.processor().lower():
            return f"amd-{machine}"
        elif "intel" in platform.processor().lower():
            return f"intel-{machine}"
        else:
            return f"{system}-{machine}"
