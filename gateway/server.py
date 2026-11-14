"""NOVA Gateway HTTP API (runs on the GPU box).

Endpoints (all need `Authorization: Bearer <token>` except /health):
  GET    /health                       engine status, roles, cloud status (no auth)
  POST   /v1/chat/completions          OpenAI-compatible chat; model = role name (brain, reflex, ...)
  POST   /v1/embeddings                OpenAI-compatible embeddings
  POST   /memory/add                   {text, kind?, importance?, source_session?}
  POST   /memory/search                {query, k?, min_score?, kinds?}
  GET    /memory?limit=&offset=&kind=  list
  PATCH  /memory/{id}                  {text?, importance?, kind?}
  DELETE /memory/{id}
  GET    /sessions?limit=&status=      list
  POST   /sessions                     {title?, channel?}
  GET    /sessions/{id}?limit=&after_id=  session + messages
  PATCH  /sessions/{id}                {title?, status?, summary?, checkpoint?}
  DELETE /sessions/{id}                delete a conversation (memories learned from it stay)
  POST   /sessions/{id}/messages       {role, content, meta?}
  GET    /gpu                          GPU temperature, power, load (for NOVA's heat limits)
  GET    /search?q=&n=                 web search via the box's SearXNG -> {results: [{title, url, snippet}]}
  POST   /stt                          raw audio body (wav/webm/ogg...) -> {text, language, duration}
                                        ?identify=1 also returns speaker: {person, score, confidence}
  GET    /voice/people                 voice recognition status + who has a voice print
  POST   /voice/people/{person}        audio body: add a voice sample;  DELETE: forget their voice
  POST   /escalate                     {messages, system?, provider?, purpose?, max_tokens?}
  GET    /escalate/usage               cloud calls per provider, last 30 days
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from nova_common.config import resolve

from .cloud import CloudError, CloudRouter
from .llm import LocalLLM
from .store import Store
from .stt import SpeechToText
from .voiceid import VoiceID

log = logging.getLogger("nova.gateway")


class AuthMiddleware:
    """Bearer-token check for every route except /health."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/health" or not self.token:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"")
        supplied = auth[7:].strip() if auth[:7].lower() == b"bearer " else b""
        if not hmac.compare_digest(supplied, self.token):
            resp = JSONResponse({"error": "unauthorized"}, status_code=401)
            return await resp(scope, receive, send)
        return await self.app(scope, receive, send)


def _err(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


async def _json(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def create_app(cfg: dict[str, Any]) -> Starlette:
    gcfg = cfg.get("gateway") or {}
    data_dir = resolve(gcfg.get("data_dir", "data/gateway"))
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(str(data_dir / "nova.db"))
    timeout = float(gcfg.get("request_timeout", 300))
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10))
    llm = LocalLLM(gcfg, client)
    cloud = CloudRouter(gcfg.get("cloud") or {}, store, client, cache_path=data_dir / "cloud_models.json")
    stt = SpeechToText(gcfg.get("stt") or {})
    voiceid = VoiceID(gcfg.get("voice_id") or {}, data_dir)
    max_audio = int((gcfg.get("stt") or {}).get("max_upload_mb", 25)) * 1024 * 1024

    # ------------------------------------------------------------ health
    async def health(request: Request):
        info = await llm.health()
        info["status"] = "ok" if info.get("engine") == "up" else "degraded"
        info["cloud"] = cloud.status()
        info["cloud_available"] = cloud.available()
        info["cloud_web"] = cloud.available_web()
        info["memories"] = store.count_memories()
        info["stt"] = stt.status()
        info["voice_id"] = voiceid.status()
        info["search"] = bool(searx_url)
        return JSONResponse(info)

    # ------------------------------------------------------------ models
    async def chat_completions(request: Request):
        body = await _json(request)
        if llm.contexts and not llm.variants:
            await llm.ensure_variants()
        if not body.get("messages"):
            return _err("messages required")
        if body.get("stream"):
            async def relay():
                try:
                    async for chunk in llm.chat_stream(body):
                        yield chunk
                except httpx.HTTPError as e:
                    log.warning("stream failed: %s", e)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
            return StreamingResponse(relay(), media_type="text/event-stream")
        try:
            return JSONResponse(await llm.chat(body))
        except (httpx.HTTPError, ValueError) as e:
            return _err(f"engine error: {e}", 502)

    async def embeddings(request: Request):
        body = await _json(request)
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            return _err("input required")
        try:
            vecs = await llm.embed(inputs, body.get("model"), body.get("purpose", ""))
        except httpx.HTTPError as e:
            return _err(f"engine error: {e}", 502)
        return JSONResponse(
            {
                "object": "list",
                "model": llm.model_for(body.get("model") or "embed"),
                "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
            }
        )

    # ------------------------------------------------------------ memory
    async def memory_add(request: Request):
        body = await _json(request)
        text = str(body.get("text") or "").strip()
        if not text:
            return _err("text required")
        try:
            [vec] = await llm.embed([text], purpose="document")
        except httpx.HTTPError as e:
            return _err(f"embedding failed: {e}", 502)
        m = store.add_memory(
            text,
            vec,
            kind=str(body.get("kind") or "fact"),
            importance=float(body.get("importance", 0.5)),
            source_session=body.get("source_session"),
            person=str(body.get("person") or ""),
            scope=str(body.get("scope") or "private"),
        )
        return JSONResponse(m)

    async def memory_search(request: Request):
        body = await _json(request)
        query = str(body.get("query") or "").strip()
        if not query:
            return _err("query required")
        try:
            [vec] = await llm.embed([query], purpose="query")
        except httpx.HTTPError as e:
            return _err(f"embedding failed: {e}", 502)
        res = store.search_memories(
            vec,
            k=int(body.get("k", 6)),
            min_score=float(body.get("min_score", 0.0)),
            kinds=body.get("kinds"),
            touch=bool(body.get("touch", True)),
            viewer=None if body.get("viewer") is None else str(body["viewer"]),
            admin=bool(body.get("admin", False)),
        )
        return JSONResponse({"results": res})

    async def memory_list(request: Request):
        q = request.query_params
        items = store.list_memories(int(q.get("limit", 100)), int(q.get("offset", 0)), q.get("kind"),
                                    viewer=q.get("viewer"), admin=q.get("admin") == "1")
        return JSONResponse({"memories": items, "total": store.count_memories()})

    async def memory_item(request: Request):
        mid = request.path_params["mid"]
        if request.method == "DELETE":
            ok = store.delete_memory(mid)
            return JSONResponse({"deleted": ok}, status_code=200 if ok else 404)
        body = await _json(request)
        vec = None
        if body.get("text"):
            try:
                [vec] = await llm.embed([str(body["text"])], purpose="document")
            except httpx.HTTPError as e:
                return _err(f"embedding failed: {e}", 502)
        m = store.update_memory(mid, body.get("text"), body.get("importance"), body.get("kind"), vec)
        return JSONResponse(m) if m else _err("not found", 404)

    # ------------------------------------------------------------ sessions
    async def sessions(request: Request):
        if request.method == "POST":
            body = await _json(request)
            return JSONResponse(store.create_session(body.get("title", "New session"), body.get("channel", "web"),
                                                     str(body.get("person") or "")))
        q = request.query_params
        return JSONResponse({"sessions": store.list_sessions(int(q.get("limit", 50)), q.get("status"), q.get("person"))})

    async def session_item(request: Request):
        sid = request.path_params["sid"]
        if store.get_session(sid) is None:
            return _err("session not found", 404)
        if request.method == "DELETE":
            return JSONResponse({"deleted": store.delete_session(sid)})
        if request.method == "PATCH":
            body = await _json(request)
            return JSONResponse(store.update_session(sid, **body))
        q = request.query_params
        s = store.get_session(sid)
        limit = int(q["limit"]) if q.get("limit") else None
        s["messages"] = store.get_messages(sid, limit=limit, after_id=int(q.get("after_id", 0)))
        s["message_count"] = store.count_messages(sid)
        return JSONResponse(s)

    async def session_messages(request: Request):
        sid = request.path_params["sid"]
        if store.get_session(sid) is None:
            return _err("session not found", 404)
        body = await _json(request)
        role = body.get("role")
        if role not in ("user", "assistant", "system", "tool"):
            return _err("role must be user, assistant, system or tool")
        return JSONResponse(store.add_message(sid, role, str(body.get("content", "")), body.get("meta")))

    # ------------------------------------------------------------ speech
    async def speech_to_text(request: Request):
        audio = await request.body()
        if not audio:
            return _err("send the audio as the request body")
        if len(audio) > max_audio:
            return _err("audio too large", 413)
        try:
            pcm = await asyncio.to_thread(stt.decode, audio)
            if request.query_params.get("identify") == "1":
                res, who = await asyncio.gather(asyncio.to_thread(stt.transcribe, pcm),
                                                asyncio.to_thread(voiceid.identify, pcm))
                res["speaker"] = who
                return JSONResponse(res)
            return JSONResponse(await asyncio.to_thread(stt.transcribe, pcm))
        except ImportError:
            return _err("faster-whisper is not installed on the GPU box (run setup_voice)", 501)
        except Exception as e:  # noqa: BLE001
            log.warning("stt failed: %s", e)
            return _err(f"speech to text failed: {e}", 500)

    async def voice_people(request: Request):
        """GET: who has a voice print. POST /voice/people/{person}: add a sample (audio body). DELETE: forget it."""
        person = request.path_params.get("person", "")
        if request.method == "GET":
            return JSONResponse(voiceid.status())
        if request.method == "DELETE":
            try:
                return JSONResponse({"deleted": voiceid.forget(person)})
            except ValueError as e:
                return _err(str(e))
        audio = await request.body()
        if not audio or len(audio) > max_audio:
            return _err("send the audio as the request body")
        if not voiceid.installed():
            return _err("Voice recognition isn't installed on the GPU box (run setup_gpu_box.sh, or pip install speechbrain)", 501)
        try:
            pcm = await asyncio.to_thread(stt.decode, audio)
            return JSONResponse(await asyncio.to_thread(voiceid.enroll, person, pcm))
        except ValueError as e:
            return _err(str(e))
        except Exception as e:  # noqa: BLE001
            log.warning("voice enrol failed: %s", e)
            return _err(f"voice enrol failed: {e}", 500)

    searx_url = str(gcfg.get("searxng_url") or "").rstrip("/")

    async def gpu(request: Request):
        """Quick GPU reading (temperature, power, load) without asking Ollama anything."""
        from .llm import gpu_info
        gpus = await asyncio.to_thread(gpu_info)
        return JSONResponse({"gpus": gpus, "max_temp_c": max((g["temp_c"] or 0 for g in gpus), default=None) if gpus else None})

    async def backup(request: Request):
        """GET -> a copy of NOVA's memory database (SQLite file), so Core can keep backups on another machine."""
        import tempfile
        from starlette.background import BackgroundTask
        from starlette.responses import FileResponse
        fd, tmp = tempfile.mkstemp(suffix=".db", dir=str(data_dir))
        os.close(fd)
        try:
            await asyncio.to_thread(store.snapshot, tmp)
        except Exception as e:  # noqa: BLE001
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            return _err(f"backup failed: {e}", 500)
        return FileResponse(tmp, media_type="application/octet-stream", filename="nova-memory.db",
                            background=BackgroundTask(lambda: os.path.exists(tmp) and os.unlink(tmp)))

    async def vision(request: Request):
        """GET ?model=brain -> {model, vision: true|false|null}: can this role's model look at images?"""
        try:
            return JSONResponse(await llm.can_see(request.query_params.get("model") or "brain"))
        except httpx.HTTPError as e:
            return _err(f"engine error: {e}", 502)

    async def web_search(request: Request):
        """Search the web through this box's private SearXNG (Google, Bing, DuckDuckGo... combined).
        GET ?q=...&n=8 -> {results: [{title, url, snippet, engines}]}. 501 if no SearXNG is configured."""
        if not searx_url:
            return _err("no search engine on this Gateway (gateway.searxng_url)", 501)
        q = (request.query_params.get("q") or "").strip()
        n = max(1, min(20, int(request.query_params.get("n", 8) or 8)))
        if not q:
            return _err("q required")
        try:
            r = await client.get(f"{searx_url}/search", params={"q": q, "format": "json", "safesearch": 0}, timeout=25)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            return _err(f"search engine error: {e}", 502)
        res = [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", ""),
                "engines": x.get("engines", [])} for x in data.get("results", []) if x.get("url")][:n]
        return JSONResponse({"results": res, "unresponsive": [e[0] for e in data.get("unresponsive_engines", [])][:6]})

    # ------------------------------------------------------------ cloud
    async def escalate(request: Request):
        body = await _json(request)
        msgs = body.get("messages") or []
        if not msgs:
            return _err("messages required")
        kw = dict(system=str(body.get("system") or ""), purpose=str(body.get("purpose") or ""), max_tokens=body.get("max_tokens"),
                  web=bool(body.get("web", True)))
        if str(body.get("provider") or "").lower() == "all":
            names = cloud.available()
            if not names:
                return _err("No cloud provider is configured yet. Run setup_cloud.bat to add your API keys.", 400)

            async def one(n):
                try:
                    return await cloud.ask(msgs, provider=n, **kw)
                except CloudError as e:
                    return {"provider": n, "model": "", "content": "", "error": str(e)}

            return JSONResponse({"provider": "all", "answers": await asyncio.gather(*(one(n) for n in names))})
        try:
            res = await cloud.ask(msgs, provider=body.get("provider"), **kw)
        except CloudError as e:
            return _err(str(e), e.status if 400 <= e.status < 600 else 502)
        return JSONResponse(res)

    async def escalate_usage(request: Request):
        return JSONResponse({"providers": cloud.status(), "last_30_days": store.usage_summary()})

    @contextlib.asynccontextmanager
    async def lifespan(app):
        log.info("Gateway ready. Data: %s", data_dir)

        async def pick_models():
            if gcfg.get("auto_pull", True):
                await llm.pull_missing()
            await llm.ensure_variants()
            await cloud.resolve_all()
            for n, st in cloud.status().items():
                if st["configured"]:
                    log.info("Cloud %s: %s%s", n, st["model"], f"  (problem: {st['error']})" if st["error"] else "")
        bg = asyncio.create_task(pick_models())
        yield
        await client.aclose()

    routes = [
        Route("/health", health),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/memory/add", memory_add, methods=["POST"]),
        Route("/memory/search", memory_search, methods=["POST"]),
        Route("/memory", memory_list),
        Route("/memory/{mid}", memory_item, methods=["PATCH", "DELETE"]),
        Route("/sessions", sessions, methods=["GET", "POST"]),
        Route("/sessions/{sid}", session_item, methods=["GET", "PATCH", "DELETE"]),
        Route("/sessions/{sid}/messages", session_messages, methods=["POST"]),
        Route("/stt", speech_to_text, methods=["POST"]),
        Route("/search", web_search),
        Route("/gpu", gpu),
        Route("/vision", vision),
        Route("/backup", backup),
        Route("/voice/people", voice_people),
        Route("/voice/people/{person}", voice_people, methods=["POST", "DELETE"]),
        Route("/escalate", escalate, methods=["POST"]),
        Route("/escalate/usage", escalate_usage),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    token = str(gcfg.get("token") or "")
    if not token or token.startswith("change-me"):
        log.warning("Gateway token is not set; the API is open to anyone on the network.")
    app.add_middleware(AuthMiddleware, token=token if not token.startswith("change-me") else "")
    app.state.store = store
    return app
