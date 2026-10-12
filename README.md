# healthcare-ai-agent
Prototype healthcare AI agent showcasing AI-driven automation, document intelligence, and conversational workflows for healthcare operations.

> **Prototype only.** All patient data in this repo is synthetic. Do not connect real PHI
> without a BAA with every vendor in the data path, durable access-controlled storage,
> audit logging, and a security review.

## What it does

| Endpoint | Capability |
|---|---|
| `POST /chat` | Front-desk scheduling agent. Claude uses tools to search patients, read records, list open slots and book appointments (after explicit confirmation). Conversations are kept server-side by `conversation_id`. |
| `POST /extract/intake` | Document intelligence. Upload a PDF or text intake form; get back validated JSON (patient, insurance, medications, allergies, urgency, missing fields). |
| `GET /health` | Liveness check. |

Built on the Anthropic Python SDK with `claude-opus-5-5`:
- **Tool use** — a manual agentic loop with `strict` tool schemas (`src/healthcare_agent/agent.py`).
- **Structured outputs** — `messages.parse` against a Pydantic model (`extraction.py`).
- **Refusal fallback** — server-side `fallbacks: "default"`; refusals map to HTTP 422.
- **Prompt caching** — automatic caching of the system prompt, tools and history.
- **PHI-aware logging** — tool inputs are redacted before logging (`phi.py`).

## Layout

```
src/healthcare_agent/
  api.py          FastAPI app and error mapping
  agent.py        scheduling agent (tool-use loop, conversation store)
  tools.py        tool schemas + dispatch
  clinic.py       in-memory clinic store (stands in for an EHR)
  extraction.py   intake-form extraction schema + call
  llm.py          shared request options, refusal handling
  phi.py          log redaction
  config.py       settings (env prefix HCA_)
  data/synthetic_clinic.json
samples/intake_form.txt
tests/            runs offline against a fake Claude client
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add ANTHROPIC_API_KEY
uvicorn healthcare_agent.api:app --reload
```

Try it:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "Find Maria Alvarez and show open primary care slots on 2026-10-14"}'

curl -s localhost:8000/extract/intake -F "file=@samples/intake_form.txt;type=text/plain"
```

Interactive API docs: http://localhost:8000/docs

## Development

```bash
ruff check . && ruff format --check .
pytest -q
```

Tests do not call the Claude API. CI runs lint and tests on Python 3.11 and 3.13.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API credentials |
| `HCA_MODEL` | `claude-opus-5-5` | Model for all calls |
| `HCA_CHAT_EFFORT` | `medium` | Effort for the chat agent |
| `HCA_EXTRACTION_EFFORT` | `high` | Effort for document extraction |
| `HCA_MAX_TOOL_ITERATIONS` | `10` | Tool-loop safety cap per turn |
| `HCA_ENABLE_REFUSAL_FALLBACK` | `true` | Server-side refusal fallback |

## Next steps

- Durable conversation + appointment storage (Postgres) with auth on the API
- Streaming chat responses for the UI
- Voice intake channel (ElevenLabs agent calling these endpoints)
- Calendar / email notifications on booking
- An eval set for extraction accuracy on synthetic forms
