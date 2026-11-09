# NOVA GPU box

The GPU half of **NOVA** (Native On-device Virtual Assistant): her models (Ollama), long-term memory and
chats, speech-to-text (Whisper) and voice recognition, behind one password-protected API on port **8800**.
NOVA's Core (the agent, web page, voice, tools) runs on another PC and talks to this box.

```
 Core PC ──(http :8800, token)──▶  nova-gateway  ──▶  nova-ollama (GPU)
                                   memory · chats ·  ──▶  nova-searxng (web search)
                                   Whisper · voice prints · cloud
```

> This repo is generated from the main NOVA folder by `publish_gpu.bat`. Edit the code there, not here.

## Install with Portainer

1. **Let Docker use the GPU** (once). On the box: `bash scripts/check_gpu.sh`. If containers can't see
   the GPU it prints the commands that install the NVIDIA Container Toolkit.
2. **Portainer → Stacks → Add stack → Repository**
   * Repository URL: `https://github.com/Sawman24/NOVA_GPU`, reference `refs/heads/main`
   * Compose path: `docker-compose.yml`
   * Private repo: turn on *Authentication* (GitHub user + a personal access token with read access)
   * Optional: *GitOps updates* (polling) so pushing to GitHub redeploys automatically
3. **Environment variables** (see `.env.example`). Only one is required:
   * `NOVA_TOKEN` = the same value as `gateway.token` / `core.gateway_token` in your PC's `nova.yaml`
   * add `GEMINI_API_KEY` etc. if you use cloud models
4. **Deploy.** The first start builds the image (a few minutes) and downloads the models
   (`bonsai-2:27b`, `bonsai-2:4b`, `nomic-embed-text`). Follow it in Portainer → Containers →
   `nova-gateway` → Logs.
5. **On the Core PC**, in `config/nova.yaml`: `core.gateway_url: http://<this-box-ip>:8800`, and
   `launcher.components: [core, voice]` (the PC stops running its own Gateway). Restart NOVA, run `doctor.bat`.

## Bring her memory over

Her memories, chats and voice prints from the PC are in `data\gateway\` there (`nova.db`, `voiceprints\`,
`cloud_models.json`). Stop NOVA on the PC, copy that folder to the box, then:

```bash
docker cp gateway/. nova-gateway:/data/      # contents of the PC's data\gateway folder
docker restart nova-gateway
```

You can also copy your PC's `nova.yaml` to `/data/nova.yaml` the same way: its `gateway:` section (token,
cloud keys, model choices) is used, and Portainer's environment variables still win.

## Day to day

| | |
|---|---|
| Logs | Portainer → `nova-gateway` → Logs |
| Health | `curl http://<box>:8800/health` |
| Pull a model by hand | `docker exec -it nova-ollama ollama pull <model>` |
| Change models | stack env `BRAIN_MODEL` / `REFLEX_MODEL` → Update the stack |
| Second V100 | set `OLLAMA_SCHED_SPREAD=1`, update the stack |
| Temperature | `watch -n2 nvidia-smi` |
| Update NOVA | run `publish_gpu.bat` on the PC, then Portainer → stack → *Pull and redeploy* (or GitOps does it) |

Data lives in Docker volumes `nova_nova-data` (`/data`) and `nova_ollama-models`. To keep them in plain
folders instead, set `NOVA_DATA=/opt/nova/data` and `OLLAMA_DIR=/opt/nova/ollama`.

Ollama and SearXNG aren't exposed to the network; only the Gateway (port 8800, token required) is.
Never commit `.env`, tokens or API keys to this repo.
