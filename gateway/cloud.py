"""Cloud escalation: Anthropic (Claude), Google (Gemini) and xAI (Grok).

Every call goes through the same guard-rails:
  * provider must have a key and a model set in config
  * per-provider daily call cap
  * privacy filter (redacts emails, phone numbers, IPs, keys and your own listed terms)
  * every call is logged to the escalations table
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from nova_common.config import secret

from .store import Store

log = logging.getLogger("nova.cloud")

ALIASES = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "grok": "grok",
    "xai": "grok",
}

MODEL_LISTS = {
    "anthropic": "https://api.anthropic.com/v1/models?limit=100",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
    "grok": "https://api.x.ai/v1/models",
}
DEFAULT_PREFER = {"anthropic": "sonnet", "gemini": "pro", "grok": ""}  # best tier first; falls back if the key can't use it
# Models that aren't general chat models.
_NOT_CHAT = re.compile(
    r"image|imagen|vision|tts|audio|speech|live|embed|veo|video|imagine|learnlm|gemma|aqa|nano|robotics|"
    r"computer-use|deep-research|code|transcribe|realtime|search|moderation|guard|lyria|antigravity|omni|"
    r"customtools|translate|native-audio|dialog", re.I)


def _version(model_id: str) -> tuple:
    """'gemini-3.1-pro-preview' -> (3, 1); 'claude-sonnet-5-5' -> (5, 5); 'grok-4-0709' -> (4,)."""
    nums = []
    for n in re.findall(r"\d+", model_id):
        v = int(n)
        if v >= 100 or len(nums) == 2:  # dates / build numbers
            break
        nums.append(v)
    return tuple(nums)


def rank_models(ids: list[str], prefer: str = "") -> list[str]:
    """Chat models, best first: preferred tier, then full-size, newest version, stable."""
    cands = [i for i in ids if i and not _NOT_CHAT.search(i)]

    def score(i: str):
        low = i.lower()
        unstable = bool(re.search(r"(?<![a-z])(preview|exp|experimental|beta|alpha)(?![a-z])", low))
        small = bool(re.search(r"(?<![a-z])(lite|mini|nano|fast|8b|haiku)(?![a-z])", low)) and prefer not in low
        return (bool(prefer) and prefer in low, not small, _version(i), not unstable, "latest" not in low, -len(i))

    return sorted(cands, key=score, reverse=True)


def choose_model(ids: list[str], prefer: str = "") -> str:
    ranked = rank_models(ids, prefer)
    return ranked[0] if ranked else ""


ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "grok": "https://api.x.ai/v1/chat/completions",
}


class CloudError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


_DATE_LIKE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")


def _phone(m: re.Match) -> str:
    s = m.group(0)
    digits = sum(c.isdigit() for c in s)
    if not 10 <= digits <= 15 or _DATE_LIKE.search(s):
        return s
    return "[phone]"


_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[ip]"),
    (re.compile(r"(?<![\w:])\+?\(?\d[\d\s().-]{8,}\d(?![\w:])"), _phone),
    (re.compile(r"\b(?:sk|xai|AIza|ghp|gho|hf)[-_A-Za-z0-9]{16,}\b"), "[secret]"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "[secret]"),
]


def _short_error(r: httpx.Response) -> str:
    """Pull the human-readable message out of a provider's error JSON (with Google's quota details)."""
    try:
        d = r.json()
        if isinstance(d, list) and d:
            d = d[0]
        e = d.get("error", d) if isinstance(d, dict) else d
        msg = e.get("message") if isinstance(e, dict) else str(e)
        msg = " ".join(str(msg).split())
        quota = []
        for det in (e.get("details") or []) if isinstance(e, dict) else []:
            for v in det.get("violations") or []:
                if v.get("quotaId"):
                    quota.append(f"{v['quotaId']} = {v.get('quotaValue', '?')}")
        if quota:
            return msg.split(" For more information")[0] + " [limit hit: " + "; ".join(dict.fromkeys(quota)) + "]"
        return msg[:300]
    except ValueError:
        return " ".join(r.text.split())[:300]


def explain_quota(name: str, errors: list[str]) -> str:
    """Plain-English summary when every model a key tried was refused for quota."""
    zero = any(re.search(r"FreeTier\S* = 0\b", x) for x in errors)
    if zero:
        return (f"{name}: Google allows this key 0 free requests on every model (free-tier limit is 0 for its project). "
                "Fix: in Google AI Studio, link a billing account to the key's project (pay-as-you-go, set a spend cap), "
                "or make a new key under a fresh AI Studio project.")
    return (f"{name}: every model refused with a quota/rate limit, so this key has used up its allowance for now. "
            "Free-tier limits reset daily; check https://aistudio.google.com/rate-limit or add billing for higher limits.")


class PrivacyFilter:
    def __init__(self, cfg: dict[str, Any]):
        self.enabled = bool(cfg.get("enabled", True))
        self.terms = [t for t in (cfg.get("redact_terms") or []) if t]

    def __call__(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        for pat, repl in _PATTERNS:
            text = pat.sub(repl, text)
        for term in self.terms:
            text = re.sub(re.escape(term), "[private]", text, flags=re.IGNORECASE)
        return text


class CloudRouter:
    def __init__(self, cfg: dict[str, Any], store: Store, client: httpx.AsyncClient, cache_path=None):
        self.cfg = cfg or {}
        self.cache_path = cache_path  # remembers the model that works for each provider
        self.store = store
        self.client = client
        self.order = [ALIASES.get(p, p) for p in (self.cfg.get("default_order") or ["anthropic", "gemini", "grok"])]
        self.privacy = PrivacyFilter(self.cfg.get("privacy") or {})
        self.providers: dict[str, dict[str, Any]] = {}
        for name in ("anthropic", "gemini", "grok"):
            p = dict(self.cfg.get(name) or {})
            model = str(p.get("model") or "").strip()
            self.providers[name] = {
                "api_key": secret(p, "api_key"),
                "model": "" if model.lower() in ("", "auto") else model,
                "auto": model.lower() in ("", "auto"),
                "prefer": str(p.get("prefer", DEFAULT_PREFER[name]) or "").lower(),
                "max_calls_per_day": int(p.get("max_calls_per_day", 50)),
                "max_tokens": int(p.get("max_tokens", 2048)),
                # Gemini (Google Search grounding) and Claude (web search tool) can look things up live.
                "web_search": bool(p.get("web_search", name in ("gemini", "anthropic"))),
            }

    async def list_models(self, name: str) -> list[str]:
        p = self.providers[name]
        if name == "anthropic":
            headers = {"x-api-key": p["api_key"], "anthropic-version": "2023-06-01"}
            r = await self.client.get(MODEL_LISTS[name], headers=headers, timeout=20)
        elif name == "gemini":
            r = await self.client.get(MODEL_LISTS[name], headers={"x-goog-api-key": p["api_key"]}, timeout=20)
        else:
            r = await self.client.get(MODEL_LISTS[name], headers={"Authorization": f"Bearer {p['api_key']}"}, timeout=20)
        if r.status_code in (401, 403):
            raise CloudError(f"{name} rejected the API key ({r.status_code}).", 401)
        if r.status_code >= 400:
            raise CloudError(f"{name} model list failed: {r.status_code} {r.text[:200]}", r.status_code)
        data = r.json()
        if name == "gemini":
            return [m["name"].removeprefix("models/") for m in data.get("models", [])
                    if "generateContent" in (m.get("supportedGenerationMethods") or [])]
        return [m["id"] for m in data.get("data", [])]

    # ---- which model each provider uses (model: auto)
    def _load_cache(self) -> dict:
        try:
            return json.loads(self.cache_path.read_text()) if self.cache_path and self.cache_path.exists() else {}
        except (OSError, ValueError):
            return {}

    def _remember(self, name: str, model: str) -> None:
        if not self.cache_path:
            return
        cache = self._load_cache()
        if cache.get(name) != model:
            cache[name] = model
            try:
                self.cache_path.write_text(json.dumps(cache, indent=2))
            except OSError:
                pass

    async def _call(self, name: str, p: dict, system: str, messages: list[dict], max_tokens: int, web: bool):
        if name == "anthropic":
            return await self._anthropic(p, system, messages, max_tokens, web and p["web_search"])
        if name == "gemini":
            return await self._gemini(p, system, messages, max_tokens, web and p["web_search"])
        return await self._openai_compatible(name, p, system, messages, max_tokens)

    async def _works(self, name: str, p: dict) -> tuple[bool, str]:
        """Tiny test call: can this key use p['model']? (A reply with no text still counts as working.)"""
        try:
            await self._call(name, p, "", [{"role": "user", "content": "Reply with OK."}], 64, False)
            return True, ""
        except CloudError as e:
            if e.status == 401:
                raise
            return e.status == 502 and "no text" in str(e), str(e)
        except httpx.HTTPError as e:
            return False, str(e)

    async def resolve(self, name: str) -> str:
        """For model: auto, find the best model this key can actually use, and remember it."""
        p = self.providers[name]
        if p["model"] or not p["api_key"]:
            return p["model"]
        ids = await self.list_models(name)
        p["candidates"] = rank_models(ids, p["prefer"])
        cached = self._load_cache().get(name)
        if cached and cached in ids:
            p["model"] = cached
            return cached
        last = "no chat models listed"
        for m in p["candidates"][:12]:
            p["model"] = m
            ok, err = await self._works(name, p)
            if ok:
                log.info("%s: using %s", name, m)
                self._remember(name, m)
                return m
            log.info("%s: %s isn't available to this key (%s)", name, m, err[:300])
            p.setdefault("skip", set()).add(m)
            if "404" not in err and "400" not in err:
                p.setdefault("errors", []).append(err)
            last = err
        p["model"] = ""
        errors = p.setdefault("errors", [])
        if errors and all("429" in x or "quota" in x.lower() for x in errors):
            raise CloudError(explain_quota(name, errors), 429)
        raise CloudError(f"{name}: none of its models work with this key yet. Last error: {last}", 429)

    async def resolve_all(self) -> None:
        for name, p in self.providers.items():
            if p["api_key"] and not p["model"]:
                try:
                    await self.resolve(name)
                except (CloudError, httpx.HTTPError) as e:
                    p["error"] = str(e)

    def status(self) -> dict[str, Any]:
        out = {}
        for name, p in self.providers.items():
            out[name] = {
                "configured": bool(p["api_key"]),
                "model": p["model"] or ("auto" if p["api_key"] else ""),
                "auto": p["auto"],
                "web_search": p["web_search"],
                "error": p.get("error", ""),
                "calls_today": self.store.escalations_today(name),
                "max_calls_per_day": p["max_calls_per_day"],
            }
        return out

    def available(self) -> list[str]:
        return [n for n in self.order if n in self.providers and self.providers[n]["api_key"] and not self.providers[n].get("error")]

    def available_web(self) -> list[str]:
        return [n for n in self.available() if self.providers[n]["web_search"]]

    def pick(self, requested: str | None) -> str:
        if requested:
            name = ALIASES.get(requested.lower().strip(), requested.lower().strip())
            if name not in self.providers:
                raise CloudError(f"Unknown provider '{requested}'. Use claude, gemini or grok.", 400)
            p = self.providers[name]
            if not p["api_key"]:
                raise CloudError(f"{name} has no API key yet. Run setup_cloud.bat.", 400)
            return name
        avail = self.available()
        if not avail:
            raise CloudError("No cloud provider is configured yet. Run setup_cloud.bat to add your API keys.", 400)
        return avail[0]

    async def ask(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        provider: str | None = None,
        purpose: str = "",
        max_tokens: int | None = None,
        web: bool = True,
    ) -> dict[str, Any]:
        if provider is None and web and self.available_web():
            provider = self.available_web()[0]
        name = self.pick(provider)
        p = self.providers[name]
        await self.resolve(name)
        if self.store.escalations_today(name) >= p["max_calls_per_day"]:
            raise CloudError(f"Daily cap reached for {name} ({p['max_calls_per_day']} calls).", 429)

        system = self.privacy(system)
        clean = [{"role": m["role"], "content": self.privacy(str(m.get("content", "")))} for m in messages]
        max_tokens = int(max_tokens or p["max_tokens"])
        tried: list[str] = []
        while True:
            try:
                content, usage = await self._call(name, p, system, clean, max_tokens, web)
                break
            except CloudError as e:
                self.store.log_escalation(name, p["model"], purpose, ok=False, error=str(e))
                # model: auto and this model stopped working (quota, retired, tier) -> next best that works.
                if p["auto"] and e.status in (400, 403, 404, 429) and len(tried) < 8:
                    tried.append(p["model"])
                    p.setdefault("skip", set()).add(p["model"])
                    if e.status == 429:
                        p.setdefault("errors", []).append(str(e))
                    nxt = [m for m in p.get("candidates", []) if m not in tried and m not in p["skip"]]
                    if nxt:
                        p["model"] = nxt[0]
                        continue
                    if e.status == 429:
                        raise CloudError(explain_quota(name, p.get("errors", [])), 429) from e
                raise
            except httpx.HTTPError as e:
                self.store.log_escalation(name, p["model"], purpose, ok=False, error=str(e))
                raise CloudError(f"{name} request failed: {e}") from e
        if p["auto"]:
            self._remember(name, p["model"])

        self.store.log_escalation(name, p["model"], purpose, usage.get("in", 0), usage.get("out", 0))
        return {"provider": name, "model": p["model"], "content": content, "usage": usage}

    async def _gemini(self, p, system, messages, max_tokens, web):
        """Gemini's native API, so Google Search grounding can be used for live questions."""
        contents: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system = f"{system}\n\n{m['content']}".strip()
                continue
            role = "model" if m["role"] == "assistant" else "user"
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"][0]["text"] += "\n\n" + m["content"]
            else:
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
        while contents and contents[0]["role"] != "user":
            contents.pop(0)
        if not contents:
            raise CloudError("Nothing to send.", 400)
        body: dict[str, Any] = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if web:
            body["tools"] = [{"google_search": {}}]
        r = await self.client.post(ENDPOINTS["gemini"].format(model=p["model"]), json=body,
                                   headers={"x-goog-api-key": p["api_key"]}, timeout=180)
        if r.status_code >= 400:
            raise CloudError(f"gemini error {r.status_code}: {_short_error(r)}", r.status_code)
        data = r.json()
        cand = (data.get("candidates") or [{}])[0]
        text = "".join(pt.get("text", "") for pt in (cand.get("content") or {}).get("parts", []) if not pt.get("thought"))
        if not text.strip():
            raise CloudError(f"gemini returned no text (finish reason: {cand.get('finishReason', 'unknown')})", 502)
        sources = []
        for ch in (cand.get("groundingMetadata") or {}).get("groundingChunks", []) or []:
            w = ch.get("web") or {}
            if w.get("uri") and w.get("title") and w["title"] not in [x[0] for x in sources]:
                sources.append((w["title"], w["uri"]))
        if sources:
            text = text.rstrip() + "\n\nSources: " + ", ".join(f"[{t}]({u})" for t, u in sources[:5])
        u = data.get("usageMetadata") or {}
        return text, {"in": u.get("promptTokenCount", 0), "out": u.get("candidatesTokenCount", 0), "web": bool(sources)}

    async def _anthropic(self, p, system, messages, max_tokens, web=False):
        # Anthropic needs alternating user/assistant turns starting with user.
        turns: list[dict[str, str]] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system = f"{system}\n\n{m['content']}".strip()
                continue
            role = "assistant" if role == "assistant" else "user"
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n\n" + m["content"]
            else:
                turns.append({"role": role, "content": m["content"]})
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        if not turns:
            raise CloudError("Nothing to send.", 400)
        body: dict[str, Any] = {"model": p["model"], "max_tokens": max_tokens, "messages": turns}
        if system:
            body["system"] = system
        if web:
            body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
        r = await self.client.post(
            ENDPOINTS["anthropic"],
            json=body,
            headers={"x-api-key": p["api_key"], "anthropic-version": "2023-06-01"},
            timeout=180,
        )
        if r.status_code >= 400:
            raise CloudError(f"anthropic error {r.status_code}: {_short_error(r)}", r.status_code)
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage") or {}
        return text, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}

    async def _openai_compatible(self, name, p, system, messages, max_tokens):
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        r = await self.client.post(
            ENDPOINTS[name],
            json={"model": p["model"], "messages": msgs, "max_tokens": max_tokens},
            headers={"Authorization": f"Bearer {p['api_key']}"},
            timeout=180,
        )
        if r.status_code >= 400:
            raise CloudError(f"{name} error {r.status_code}: {_short_error(r)}", r.status_code)
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        u = data.get("usage") or {}
        return text, {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
