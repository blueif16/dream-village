# Dream Village inspector

A read-only browser for the Dream Village (full13-v1), its v0 memory replay and the scorer's outputs. Static: the app
only fetches the JSON bundle that `export.py` writes to `public/data/`.

From the repo root:

```bash
.venv/bin/python viz/export.py          # write viz/public/data/ (re-run as scorer files appear)
cd viz && npm install
npm run dev                             # http://localhost:5173 (Town view: http://localhost:5173/#/town)
```

`export.py --village --replay --scorer --out` override the default paths (the main checkout's `data/villages/full13-v1`,
`runs/replays/full13-v1-v0`, `runs/scorer/full13-v1-v0`).
