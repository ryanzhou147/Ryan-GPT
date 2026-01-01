from pathlib import Path
import threading
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
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
    # Map logical kinds to the models directory
    models_dir = Path('models')
    mapping = {
        'finetune': models_dir / 'finetune_dailydialog' / 'ckpt_final.pt',
        'pretrain': models_dir / 'pretrain_wikipedia' / 'ckpt_final.pt',
    }

    # If `kind` is a direct checkpoint path, prefer that
    p = Path(kind)
    if p.suffix == '.pt' and p.exists():
        resolved_checkpoint = str(p)
        inferred = None
    else:
        key_lower = 'finetune' if 'finetune' in str(kind).lower() else 'pretrain'
        resolved_checkpoint = str(mapping.get(key_lower))
        inferred = key_lower

    cache_key = f"checkpoint:{resolved_checkpoint}"
    with _LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        if not Path(resolved_checkpoint).exists():
            raise RuntimeError(f"checkpoint not found: {resolved_checkpoint}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # choose tokenizer preset based on logical kind or inferred from path
        if inferred:
            preset = PRESETS.get('chat') if inferred == 'finetune' else PRESETS.get('wikipedia')
        else:
            lc = str(resolved_checkpoint).lower()
            if 'finetune' in lc:
                preset = PRESETS.get('chat')
            else:
                preset = PRESETS.get('wikipedia')
        model = load_model(resolved_checkpoint, device)
        tokenizer = load_tokenizer(preset['vocab'], preset['merges'])
        _MODEL_CACHE[cache_key] = (model, tokenizer, device)
        return _MODEL_CACHE[cache_key]


def list_available_models():
    """Scan `models/` for subfolders with `ckpt_final.pt` and return them.

    Also include preset fallbacks if any expected models are missing.
    """
    models_dir = Path('models')
    results = []
    if not models_dir.exists():
        # fallback to presets
        if 'chat' in PRESETS:
            results.append({'id': 'finetune', 'label': 'finetune (preset)', 'checkpoint': PRESETS['chat'].get('checkpoint')})
        if 'wikipedia' in PRESETS:
            results.append({'id': 'pretrain', 'label': 'pretrain (preset)', 'checkpoint': PRESETS['wikipedia'].get('checkpoint')})
        return results

    for sub in sorted([p for p in models_dir.iterdir() if p.is_dir()]):
        ck = sub / 'ckpt_final.pt'
        if ck.exists():
            results.append({'id': sub.name, 'label': f"{sub.name}", 'checkpoint': str(ck)})

    # ensure logical finetune/pretrain entries exist (fallback to presets if not)
    has_finetune = any('finetune' in r['id'].lower() for r in results)
    has_pretrain = any('pretrain' in r['id'].lower() for r in results)
    if not has_finetune and 'chat' in PRESETS:
        results.append({'id': 'finetune', 'label': 'finetune (preset)', 'checkpoint': PRESETS['chat'].get('checkpoint')})
    if not has_pretrain and 'wikipedia' in PRESETS:
        results.append({'id': 'pretrain', 'label': 'pretrain (preset)', 'checkpoint': PRESETS['wikipedia'].get('checkpoint')})

    return results


@app.get('/api/models')
async def api_models():
    models = list_available_models()
    return JSONResponse({'models': models})

@app.get('/favicon.ico')
async def favicon():
    return Response(status_code=204)  # No content

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

    # Determine chat mode first
    chat_mode = 'finetune' in model_type.lower()
    
    # Set defaults and bounds based on model type
    if chat_mode:
        max_tokens = 50
        default_min = 5
        min_bound = 5
    else:
        max_tokens = 75
        default_min = 10
        min_bound = 10
    
    # Apply min_tokens bounds
    if min_tokens < min_bound:
        min_tokens = min_bound
    if min_tokens > max_tokens - 10:  # Leave room for generation
        min_tokens = max_tokens - 10

    # Temperature bounds
    if temperature <= 0.0:
        temperature = 0.01
    if temperature >= 1.0:
        temperature = 0.99
    temperature = round(temperature, 2)

    try:
        model, tokenizer, device = get_model_and_tokenizer(model_type)
        reply = generate_response(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
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
