#!/usr/bin/env python3
from pathlib import Path
import threading
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ryan_gpt_basics.generate import (
    load_model,
    load_tokenizer,
    generate_response,
    PRESETS,
)

app = FastAPI(title="RyanGPT Web UI")

# Mount static files directory
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Allow basic CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for loaded models/tokenizers
_MODEL_CACHE = {}
_LOCK = threading.Lock()


def get_model_and_tokenizer(kind: str):
    """kind: 'finetune' or 'pretrain'"""
    # Resolve to a concrete checkpoint path (prefer the newest run for this kind)
    def find_latest_checkpoint(kind_name: str):
        runs_dir = Path('runs')
        if not runs_dir.exists():
            return None
        candidates = []
        for p in runs_dir.glob('**/checkpoints/*.pt'):
            # prefer checkpoints in runs/ directories that include the kind name
            if kind_name.lower() in str(p).lower():
                candidates.append(p)
        # if none matched by name, fall back to any checkpoint
        if not candidates:
            candidates = list(runs_dir.glob('**/checkpoints/*.pt'))
        if not candidates:
            return None
        # pick the most recently modified checkpoint
        latest = max(candidates, key=lambda x: x.stat().st_mtime)
        return str(latest)

    # If `kind` looks like a direct checkpoint path, use it directly
    p = Path(kind)
    if p.suffix == '.pt' and p.exists():
        resolved_checkpoint = str(p)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cache_key = f"checkpoint:{resolved_checkpoint}"
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        # heuristic: choose tokenizer preset based on run name
        preset_key = 'chat' if 'finetune' in str(p).lower() else 'wikipedia'
        preset = PRESETS.get(preset_key, PRESETS.get('wikipedia'))
        model = load_model(resolved_checkpoint, device)
        tokenizer = load_tokenizer(preset['vocab'], preset['merges'])
        _MODEL_CACHE[cache_key] = (model, tokenizer, device)
        return _MODEL_CACHE[cache_key]

    # Determine the checkpoint path to use and include it in the cache key
    key = kind
    with _LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        preset = PRESETS['chat'] if kind == 'finetune' else PRESETS['wikipedia']
        # try to find a more recent checkpoint in runs/ for this kind
        resolved_checkpoint = find_latest_checkpoint(kind) or preset.get('checkpoint')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # use checkpoint path as part of the cache key so different checkpoints are cached separately
        cache_key = f"{kind}:{resolved_checkpoint}"
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        if not resolved_checkpoint:
            raise RuntimeError(f"no checkpoint found for model kind '{kind}' and no preset available")

        model = load_model(resolved_checkpoint, device)
        tokenizer = load_tokenizer(preset['vocab'], preset['merges'])
        _MODEL_CACHE[cache_key] = (model, tokenizer, device)
        return _MODEL_CACHE[cache_key]


def list_available_models():
    """Scan `runs/*/checkpoints/*.pt` and return the newest checkpoint for each run folder."""
    runs_dir = Path('runs')
    results = []
    if not runs_dir.exists():
        return results

    # consider each immediate child under runs/ as a model-run folder
    for run_sub in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        chk_dir = run_sub / 'checkpoints'
        if not chk_dir.exists():
            continue
        cands = list(chk_dir.glob('*.pt'))
        if not cands:
            continue
        latest = max(cands, key=lambda x: x.stat().st_mtime)
        results.append({
            'id': run_sub.name,
            'label': f"{run_sub.name} ({latest.name})",
            'checkpoint': str(latest),
        })

    # also include presets if not already present
    for k, v in PRESETS.items():
        # use run-like id for preset
        rid = k
        if any(r['id'] == rid for r in results):
            continue
        results.append({'id': rid, 'label': f"preset: {rid}", 'checkpoint': v.get('checkpoint')})

    return results


@app.get('/api/models')
async def api_models():
    models = list_available_models()
    return JSONResponse({'models': models})


@app.get('/')
async def index():
    return FileResponse('webapp/templates/index.html')

@app.post('/api/chat')
async def api_chat(req: Request):
    body = await req.json()
    model_type = body.get('model', 'finetune')
    prompt = (body.get('prompt') or '').strip()
    try:
        min_tokens = int(body.get('min_tokens', 10))
    except Exception:
        min_tokens = 10
    try:
        temperature = float(body.get('temperature', 0.5))
    except Exception:
        temperature = 0.5

    if not prompt:
        raise HTTPException(status_code=400, detail='empty prompt')

    # enforce bounds: temperature in (0,1) exclusive, min_tokens in [10,140]
    if temperature <= 0.0:
        temperature = 0.01
    if temperature >= 1.0:
        temperature = 0.99

    # round temperature to 2 decimal places server-side
    try:
        temperature = float(round(float(temperature), 2))
    except Exception:
        temperature = 0.5

    if min_tokens < 10:
        min_tokens = 10
    if min_tokens > 140:
        min_tokens = 140
    # ensure min_tokens is integer and keep it within 3 digits
    try:
        min_tokens = int(min_tokens)
    except Exception:
        min_tokens = 10
    if min_tokens > 999:
        min_tokens = 999

    chat_mode = 'finetune' in model_type.lower()

    try:
        model, tokenizer, device = get_model_and_tokenizer(model_type)
        reply = generate_response(
            model, tokenizer, prompt,
            max_tokens=150,
            min_tokens=min_tokens,
            temperature=temperature,
            device=device,
            chat_mode=chat_mode,
        )
        return JSONResponse({'reply': reply})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    print(f"Starting RyanGPT web UI on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
