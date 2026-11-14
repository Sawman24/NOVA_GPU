"""Local model access. Talks to any OpenAI-compatible engine (Ollama by default).

Callers use *role names* (brain, reflex, embed, ...) instead of model names, so
swapping a model is a one-line config change.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger("nova.llm")


class LocalLLM:
    def __init__(self, cfg: dict[str, Any], client: httpx.AsyncClient):
        self.base = str(cfg.get("ollama_url", "http://localhost:11434")).rstrip("/")
        self.roles: dict[str, str] = dict(cfg.get("roles") or {})
        self.options: dict[str, Any] = dict(cfg.get("model_options") or {})
        self.embed_query_prefix = cfg.get("embed_query_prefix", "")
        self.embed_doc_prefix = cfg.get("embed_doc_prefix", "")
        self.client = client
        # Per-role context window (tokens). Ollama's default is small; long thinking + tools need room.
        self.contexts: dict[str, int] = {k: int(v) for k, v in (cfg.get("context") or {"brain": 24576}).items() if v}
        self.variants: dict[str, str] = {}  # role -> Ollama model created with the bigger context
        # Some models ship with a small output cap (num_predict). -1 = no cap beyond the context window.
        self.max_output = int(cfg.get("max_output_tokens", -1))
        # Take a role's model off the GPU once it has been idle this many seconds (0 / missing = keep it loaded).
        # The small reflex model otherwise sits in VRAM next to the Brain all day; it reloads in a few seconds.
        self.unload_after: dict[str, float] = {k: float(v) for k, v in (cfg.get("unload_after") or {"reflex": 30}).items() if v}
        self._busy: dict[str, int] = {}      # role -> requests running now
        self._last_use: dict[str, float] = {}
        self._unloaders: dict[str, asyncio.Task] = {}

    def model_for(self, name: str | None) -> str:
        name = name or "brain"
        return self.variants.get(name) or self.roles.get(name, name)

    @staticmethod
    def variant_name(base: str, ctx: int) -> str:
        return re.sub(r"[^a-z0-9.-]+", "-", base.lower()).strip("-") + f"-nova{ctx // 1024}k"

    async def pull_missing(self) -> list[str]:
        """Download any role model Ollama doesn't have yet (first start on a new box). Waits for Ollama to come up."""
        installed: set = set()
        for attempt in range(60):
            try:
                r = await self.client.get(f"{self.base}/api/tags", timeout=5)
                installed = {m.get("name") or m.get("model") for m in r.json().get("models", [])}
                break
            except Exception:  # noqa: BLE001
                await asyncio.sleep(5)
        else:
            log.warning("Ollama didn't answer at %s; can't check models", self.base)
            return []
        missing = [m for m in dict.fromkeys(self.roles.values()) if m and m not in installed and f"{m}:latest" not in installed]
        pulled = []
        for m in missing:
            log.info("Downloading model %s (first start; this can take a while)...", m)
            self.pulling = m
            try:
                last = 0.0
                async with self.client.stream("POST", f"{self.base}/api/pull", json={"model": m, "name": m, "stream": True},
                                              timeout=None) as r:
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        if ev.get("error"):
                            raise RuntimeError(ev["error"])
                        total, done = ev.get("total") or 0, ev.get("completed") or 0
                        pct = 100 * done / total if total else 0
                        if total and pct - last >= 10:
                            last = pct
                            log.info("  %s: %d%% of %.1f GB", m, pct, total / 1e9)
                log.info("Downloaded %s", m)
                pulled.append(m)
            except Exception as e:  # noqa: BLE001
                log.warning("Couldn't download %s: %s. Pull it by hand: docker exec -it nova-ollama ollama pull %s", m, e, m)
            finally:
                self.pulling = ""
        if pulled:
            self._variants_tried = 0  # make the bigger-context copies now that the models exist
        return pulled

    async def ensure_variants(self) -> None:
        """Make Ollama copies of the role models with a bigger context window (instant; no extra disk).
        Falls back to the plain model if this Ollama can't create them."""
        import time as _t

        if _t.time() - getattr(self, "_variants_tried", 0) < 300 and getattr(self, "_variants_tried", 0):
            return
        self._variants_tried = _t.time()
        try:
            r = await self.client.get(f"{self.base}/api/tags", timeout=5)
            installed = {m.get("name") or m.get("model") for m in r.json().get("models", [])}
        except Exception as e:  # noqa: BLE001
            log.info("Can't check models yet (%s); will retry", e)
            return
        have = lambda n: n in installed or f"{n}:latest" in installed  # noqa: E731
        for role, ctx in self.contexts.items():
            base = self.roles.get(role)
            if not base or not have(base):
                continue
            name = self.variant_name(base, ctx)
            if True:  # (re)create every start so parameter changes always apply; it's instant
                params = {"num_ctx": ctx, "num_predict": self.max_output}
                body = {"model": name, "from": base, "parameters": params, "stream": False}
                try:
                    c = await self.client.post(f"{self.base}/api/create", json=body, timeout=120)
                    if c.status_code >= 400:  # older Ollama: Modelfile form
                        c = await self.client.post(f"{self.base}/api/create", timeout=120, json={
                            "name": name, "modelfile": f"FROM {base}\nPARAMETER num_ctx {ctx}\nPARAMETER num_predict {self.max_output}",
                            "stream": False})
                    c.raise_for_status()
                    log.info("Ready: %s (%s, %d-token context, output cap %s) for the %s role", name, base, ctx,
                             "none" if self.max_output < 0 else self.max_output, role)
                except Exception as e:  # noqa: BLE001
                    log.warning("Couldn't create a %d-token version of %s (%s); using its default context", ctx, base, e)
                    continue
            self.variants[role] = name

    def prepare_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        role = body.get("model") or "brain"
        body["model"] = self.model_for(role)
        # Per-role defaults from config (e.g. temperature), without overriding the caller.
        for k, v in (self.options.get(role) or {}).items():
            body.setdefault(k, v)
        if "max_tokens" not in body and role in self.contexts:
            # Ollama turns max_tokens into num_predict; without it some models stop at their built-in cap.
            body["max_tokens"] = self.contexts[role] if self.max_output < 0 else self.max_output
        return body

    async def can_see(self, role: str = "brain") -> dict[str, Any]:
        """Does the model behind this role accept images? Ollama lists 'vision' in its capabilities."""
        base = self.roles.get(role, role)
        r = await self.client.post(f"{self.base}/api/show", json={"model": base}, timeout=15)
        r.raise_for_status()
        caps = r.json().get("capabilities")
        # Older Ollama versions don't report capabilities: unknown (None), let the caller try.
        return {"model": base, "vision": ("vision" in caps) if isinstance(caps, list) else None, "capabilities": caps}

    # ------------------------------------------------------------------ freeing VRAM
    def _begin(self, role: str) -> None:
        if role in self.unload_after:
            self._busy[role] = self._busy.get(role, 0) + 1

    def _end(self, role: str) -> None:
        if role not in self.unload_after:
            return
        self._busy[role] = max(0, self._busy.get(role, 0) - 1)
        self._last_use[role] = asyncio.get_running_loop().time()
        t = self._unloaders.get(role)
        if t is None or t.done():
            self._unloaders[role] = asyncio.create_task(self._unload_when_idle(role))

    async def _unload_when_idle(self, role: str) -> None:
        wait = self.unload_after[role]
        loop = asyncio.get_running_loop()
        while True:
            idle = loop.time() - self._last_use.get(role, 0)
            if self._busy.get(role, 0) == 0 and idle >= wait:
                break
            await asyncio.sleep(max(0.5, wait - idle) if self._busy.get(role, 0) == 0 else 1.0)
        model = self.model_for(role)
        if any(self.model_for(r) == model for r in self.roles if r != role and r not in self.unload_after):
            return  # another role that stays loaded uses the same model
        try:  # an empty request with keep_alive 0 is Ollama's "unload this model now"
            r = await self.client.post(f"{self.base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
            r.raise_for_status()
            log.info("Unloaded %s (%s role idle for %ds): VRAM freed", model, role, wait)
        except Exception as e:  # noqa: BLE001
            log.warning("Couldn't unload %s: %s", model, e)

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        role = body.get("model") or "brain"
        self._begin(role)
        try:
            return await self._chat(body)
        finally:
            self._end(role)

    async def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        body = self.prepare_chat(body)
        body["stream"] = False
        for attempt in range(2):
            r = await self.client.post(f"{self.base}/v1/chat/completions", json=body)
            if attempt == 0 and r.status_code >= 500 and _runner_crashed(r.text):
                log.warning("Ollama's model runner crashed; retrying once")
                await asyncio.sleep(3)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()

    async def chat_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Relay the engine's server-sent events unchanged. If Ollama's model runner crashed before
        answering (it happens under memory pressure), it's restarted by the next request: retry once."""
        role = body.get("model") or "brain"
        self._begin(role)
        try:
            async for chunk in self._chat_stream(body):
                yield chunk
        finally:
            self._end(role)

    async def _chat_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        body = self.prepare_chat(body)
        body["stream"] = True
        for attempt in range(2):
            async with self.client.stream("POST", f"{self.base}/v1/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    detail = (await r.aread()).decode("utf-8", "replace")[:500]
                    if attempt == 0 and r.status_code >= 500 and _runner_crashed(detail):
                        log.warning("Ollama's model runner crashed (%s); retrying once", detail[:120])
                        await asyncio.sleep(3)
                        continue
                    raise httpx.HTTPStatusError(f"engine error {r.status_code}: {detail}", request=r.request, response=r)
                async for chunk in r.aiter_raw():
                    yield chunk
                return

    async def embed(self, texts: list[str], model: str | None = None, purpose: str = "") -> list[list[float]]:
        """purpose: 'query' or 'document' adds the embedder's prefix (nomic-embed-text wants them)."""
        prefix = {"query": self.embed_query_prefix, "document": self.embed_doc_prefix}.get(purpose, "")
        inputs = [prefix + t for t in texts]
        m = self.model_for(model or "embed")
        r = await self.client.post(f"{self.base}/v1/embeddings", json={"model": m, "input": inputs})
        if r.status_code == 404:  # older Ollama: native endpoint
            r = await self.client.post(f"{self.base}/api/embed", json={"model": m, "input": inputs})
            r.raise_for_status()
            return r.json()["embeddings"]
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    async def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {"engine_url": self.base, "roles": self.roles}
        try:
            r = await self.client.get(f"{self.base}/api/tags", timeout=5)
            r.raise_for_status()
            installed = {m.get("name") or m.get("model") for m in r.json().get("models", [])}
            info["engine"] = "up"
            info["installed_models"] = sorted(x for x in installed if x)

            def present(name: str) -> bool:
                return name in installed or f"{name}:latest" in installed

            info["missing_models"] = sorted({m for m in self.roles.values() if not present(m)})
            if getattr(self, "pulling", ""):
                info["downloading"] = self.pulling
        except Exception as e:  # noqa: BLE001
            info["engine"] = "down"
            info["error"] = str(e)
        info["context"] = {role: {"tokens": ctx, "model": self.variants.get(role, "(default context)")}
                           for role, ctx in self.contexts.items()}
        info["loaded"] = await self.loaded()
        info["gpus"] = await asyncio.to_thread(gpu_info)
        return info

    async def loaded(self) -> list[dict[str, Any]]:
        """Models Ollama has in memory right now, and how much of each is on the GPU."""
        try:
            r = await self.client.get(f"{self.base}/api/ps", timeout=5)
            r.raise_for_status()
        except Exception:  # noqa: BLE001
            return []
        out = []
        for m in r.json().get("models", []):
            size, vram = int(m.get("size") or 0), int(m.get("size_vram") or 0)
            out.append({"name": m.get("name") or m.get("model"), "size_gb": round(size / 1e9, 1),
                        "gpu_percent": round(100 * vram / size) if size else 0,
                        "context": m.get("context_length")})
        return out


def _runner_crashed(detail: str) -> bool:
    d = detail.lower()
    signs = ("runner has unexpectedly stopped", "runner process has terminated", "connection refused",
             "unexpected eof", "connection reset", "broken pipe", "error was encountered while running the model")
    return any(s in d for s in signs)


def gpu_info() -> list[dict[str, Any]]:
    """GPUs on this machine (nvidia-smi), or [] if it isn't available."""
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw,power.limit",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            def num(x):
                try:
                    return int(float(x))
                except ValueError:
                    return None
            gpus.append({"name": parts[0], "used_mb": num(parts[1]), "total_mb": num(parts[2]),
                         "temp_c": num(parts[3]) if len(parts) > 3 else None,
                         "util_percent": num(parts[4]) if len(parts) > 4 else None,
                         "power_w": num(parts[5]) if len(parts) > 5 else None,
                         "power_limit_w": num(parts[6]) if len(parts) > 6 else None})
    return gpus
