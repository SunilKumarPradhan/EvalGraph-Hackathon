# EvalGraph

Analyze any LLM application repo, map its prompt/flow structure, and run evaluations against it.

## Quick start

See `backend/pyproject.toml` and `frontend/package.json` for setup. Backend runs on FastAPI (`uvicorn evalgraph.api.main:app`), frontend on Vite.

Set `EVALGRAPH_MOCK_LLM=1` to run without a live LLM key.
