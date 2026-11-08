"""Container start: build the Gateway's config from defaults + /data/nova.yaml + environment variables,
point CUDA at the pip-installed cuBLAS/cuDNN, then run the Gateway."""
from __future__ import annotations

import os
import pathlib
import secrets
import sys

import yaml

APP = pathlib.Path("/app")
DATA = pathlib.Path(os.environ.get("NOVA_DATA_DIR", "/data"))


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((APP / "docker" / "gateway.yaml").read_text(encoding="utf-8")) or {}
    user = DATA / "nova.yaml"
    if user.exists():  # e.g. your PC's nova.yaml copied in: only its gateway section is used
        cfg = merge(cfg, {"gateway": (yaml.safe_load(user.read_text(encoding="utf-8")) or {}).get("gateway") or {}})
    g = cfg["gateway"]
    g["data_dir"] = str(DATA)
    g["host"], g["port"] = "0.0.0.0", 8800

    if env("OLLAMA_URL"):
        g["ollama_url"] = env("OLLAMA_URL")
    if env("SEARXNG_URL"):
        g["searxng_url"] = env("SEARXNG_URL")
    for role, var in (("brain", "BRAIN_MODEL"), ("reflex", "REFLEX_MODEL"), ("embed", "EMBED_MODEL")):
        if env(var):
            g.setdefault("roles", {})[role] = env(var)
    for role, var in (("brain", "BRAIN_CONTEXT"), ("reflex", "REFLEX_CONTEXT")):
        if env(var).isdigit():
            g.setdefault("context", {})[role] = int(env(var))
    if env("WHISPER_MODEL"):
        g.setdefault("stt", {})["model"] = env("WHISPER_MODEL")
    if env("WHISPER_DEVICE"):
        g.setdefault("stt", {})["device"] = env("WHISPER_DEVICE")

    token = env("NOVA_TOKEN") or str(g.get("token") or "")
    if not token or token == "change-me":
        tf = DATA / "token.txt"
        token = tf.read_text().strip() if tf.exists() else secrets.token_urlsafe(24)
        tf.write_text(token)
        bar = "=" * 72
        print(f"\n{bar}\nNOVA_TOKEN isn't set, so a password was generated (saved in the data volume, token.txt):\n\n"
              f"    {token}\n\nPut it in nova.yaml on the Core PC as core.gateway_token, or set NOVA_TOKEN in Portainer "
              f"to your existing one.\n{bar}\n", flush=True)
    g["token"] = token

    run_cfg = DATA / ".gateway-config.yaml"
    run_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    try:
        run_cfg.chmod(0o600)
    except OSError:
        pass

    # faster-whisper loads cuBLAS/cuDNN with dlopen: they must be on LD_LIBRARY_PATH when the process starts.
    libs = []
    for mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            m = __import__(mod, fromlist=["_"])
            libs.append(os.path.dirname(m.__file__) if getattr(m, "__file__", None) else list(m.__path__)[0])
        except Exception:  # noqa: BLE001
            pass
    run_env = dict(os.environ, NOVA_CONFIG=str(run_cfg), PYTHONPATH=str(APP))
    if libs:
        run_env["LD_LIBRARY_PATH"] = ":".join(libs + [run_env.get("LD_LIBRARY_PATH", "")]).strip(":")
    roles = g.get("roles", {})
    print(f"NOVA Gateway: brain={roles.get('brain')} reflex={roles.get('reflex')} embed={roles.get('embed')} "
          f"ollama={g['ollama_url']} data={DATA}", flush=True)
    os.chdir(APP)
    os.execvpe(sys.executable, [sys.executable, "-m", "gateway"], run_env)


if __name__ == "__main__":
    main()
