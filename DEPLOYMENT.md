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

## Option B — Hugging Face Spaces (free, public URL) ✅ RECOMMENDED

Free CPU tier (2 vCPU / 16 GB), permanent public URL, well-recognized
platform. Ollama runs inside the app container; the LLM is baked into
the image at build time so restarts don't re-download 2 GB.

**Ready-to-use files:** `deploy/hf/` contains the Space Dockerfile,
`start.sh`, and the Space `README.md` front-matter.

Steps:

1. Create a free account at https://huggingface.co
2. Create a write token: Settings → Access Tokens → New (type: Write)
3. Choose one:
   - Give the token to your assistant → the Space is created and
     pushed automatically (repo type: space, SDK: docker)
   - Or manually: create a new Space (SDK = Docker, CPU basic),
     clone it, copy the repo contents + `deploy/hf/*` files in
     (`deploy/hf/SPACE_README.md` becomes the Space's `README.md`),
     and push.

Notes:
- Free Spaces sleep after 48 h of inactivity and wake on the first
  visit (first answer after a wake is slow — the model warms up).
- Baked image ≈ 4 GB; build takes ~5–10 min the first time.
- CPU-only inference: expect ~30–60 s per answer. Fine for a demo.

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