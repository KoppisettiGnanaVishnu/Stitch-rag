# Deploying Stitch

Stitch is fully containerised (`Dockerfile` + `docker-compose.yml`).
Pick the path that matches your budget and audience.

## Option A — One-command VPS (recommended, ~$5–10/mo)

Works on Hetzner, DigitalOcean, Linode, etc. Ollama runs in the
same box, so the "fully local" story stays intact.

```bash
# On the server
git clone https://github.com/KoppisettiGnanaVishnu/Stitch-rag.git
cd Stitch-rag
docker compose up -d
```

Then put Caddy or nginx in front for HTTPS:

```bash
# /etc/caddy/Caddyfile
stitch.yourdomain.com {
    reverse_proxy localhost:8501
}
```

Resource note: `llama3.2:3b` needs ~4 GB RAM. A 4 vCPU / 8 GB
instance is the comfortable minimum. First start pulls the model
(~2 GB) — allow 5 minutes.

## Option B — Hugging Face Spaces (free, public URL)

HF Spaces runs Docker images on CPU. Create a new Space with
SDK = Docker, then push this repo. The Space will build the
Dockerfile and expose port 7860 — add to the Dockerfile:

```dockerfile
ENV STREAMLIT_SERVER_PORT=7860
```

Caveat: CPU-only inference for `llama3.2:3b` is slow (~30–60 s
per answer). Fine for a demo link, not for real use. Consider
switching `STITCH_GENERATOR_MODEL` to a smaller model or a
hosted endpoint for the demo.

## Option C — Render / Fly.io (managed)

Both support Docker. Render free tier cannot run Ollama
in-process (RAM limits), so point `STITCH_OLLAMA_URL` at a
hosted LLM endpoint (OpenAI-compatible) or run Ollama on a
separate box. `render.yaml` is included as a starting point.

## What I need from you to deploy now

Live deployment requires your cloud account and credentials,
which I cannot create on your behalf. Tell me which option you
want and provide:

- Option A: server IP + SSH access (I'll write the bootstrap script)
- Option B: your Hugging Face username + a write token
- Option C: your Render/Fly account + API token

Until then, the repo is deployment-ready: `docker compose up -d`
is the entire runbook.