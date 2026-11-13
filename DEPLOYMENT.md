# Deploying Stitch

Stitch is fully containerised (`Dockerfile` + `docker-compose.yml`).
Pick the path that matches your budget and audience.

## Option A — Oracle Cloud Always Free ✅ BEST FREE OPTION

Genuinely free forever, no monthly bill: up to **4 ARM vCPUs +
24 GB RAM** (Ampere A1) plus 200 GB storage. Comfortably runs the
whole `docker compose` stack (Streamlit ~2 GB + Ollama 3B ~4 GB).

```bash
# On the server
git clone https://github.com/KoppisettiGnanaVishnu/Stitch-rag.git
cd Stitch-rag
docker compose up -d
```

Requirements:
- Credit card at signup (identity check only — $0 charged on
  the Always Free tier)
- Pick an ARM (Ampere A1) instance; free-tier capacity can be
  scarce in popular regions — retry or pick a less busy region
- Then put Caddy in front for HTTPS:

```bash
# /etc/caddy/Caddyfile
stitch.yourdomain.com {
    reverse_proxy localhost:8501
}
```

## Option C — Render / Fly.io (managed, paid)

Both support Docker. Render free tier cannot run Ollama
in-process (RAM limits), so point `STITCH_OLLAMA_URL` at a
hosted LLM endpoint (OpenAI-compatible) or run Ollama on a
separate box. `render.yaml` is included as a starting point.

## What I need from you to deploy now

- **Oracle Always Free (recommended):** create the account, launch
  an Ampere A1 instance (Ubuntu 22.04+), give me the public IP and
  SSH key access — I'll bootstrap Docker + the stack + HTTPS.
- **HF PRO ($9/mo):** subscribe, tell me, and the Space goes live
  from `deploy/hf/` in minutes.
- **Your own VPS:** any Linux box works — same one-command runbook.