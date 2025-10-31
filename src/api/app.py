# src/api/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import io
import uuid
import sqlite3
from datetime import datetime, timedelta, timezone
import json
import time
import hashlib
import torch
from PIL import Image, ImageFilter
import torchvision.transforms as T
import uvicorn
import tempfile
import mimetypes
from pydantic import BaseModel

from src.models.model_factory import create_model
from src.preprocess_images import get_transforms  # uses the transforms you implemented

# Config (allow override via env MODEL_PATH)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/regularized_training/model_swin_t_regularized.pth"))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else (
         torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))

# App data paths
APP_DATA_DIR = Path("web_app_data")
UPLOAD_DIR = APP_DATA_DIR / "uploads"
DB_PATH = APP_DATA_DIR / "app.db"

app = FastAPI(title="AI Image Detector - Inference API")

# Allow CORS from local dev frontend (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:5001",
        "http://127.0.0.1:5001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure storage directories exist and mount static files
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(UPLOAD_DIR)), name="images")

# Load model once at startup
MODEL_NAME = os.environ.get("MODEL_NAME", "swin_tiny_patch4_window7_224")
NUM_CLASSES = 2
CLASS_NAMES = ["ai", "nature"]
TEMPERATURE_PATH = Path(os.environ.get("TEMPERATURE_PATH", "results/temperature_swin_tiny_patch4_window7_224.txt"))
_TEMPERATURE: float | None = None
GENERATOR_HEAD_PATH = os.environ.get("GENERATOR_HEAD_PATH", "")
GENERATOR_LABELS = [s.strip() for s in os.environ.get(
    "GENERATOR_LABELS",
    "adm,biggan,glide,midjourney,sdv5,vqdm,wukong"
).split(",") if s.strip()]

# Decision thresholding (to mitigate crisp-real false positives)
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

THRESHOLD_AI: float = _env_float("THRESHOLD_AI", 0.5)
THRESHOLD_AI_ADAPT_K: float = _env_float("THRESHOLD_AI_ADAPT_K", 0.0)
THRESHOLD_AI_ADAPT_FLOOR: float = _env_float("THRESHOLD_AI_ADAPT_FLOOR", 0.02)

def load_model(path: Path, model_name: str = MODEL_NAME):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Infer if head is Sequential(Dropout, Linear) by looking for 'fc.1.*' keys
    use_seq_head = any(k.startswith("fc.1.") for k in state_dict.keys())
    inferred_dropout = 0.5 if use_seq_head else 0.0
    model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=False, dropout=inferred_dropout)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = None
val_transform = None
MODEL_VERSION: dict[str, str] | None = None
_feat_buf: dict[str, torch.Tensor] = {}
_gen_head: torch.nn.Linear | None = None
_gen_proto: dict[str, torch.Tensor] | None = None  # label -> feature prototype (1D tensor on DEVICE)

# --- SQLite helpers ---
def get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                mime_type TEXT,
                width INTEGER,
                height INTEGER,
                prediction TEXT NOT NULL,
                prob_ai REAL NOT NULL,
                prob_nature REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT,
                inference_time_ms REAL,
                model_version TEXT,
                ground_truth_label TEXT,
                generator_label TEXT,
                generator_json TEXT,
                file_size INTEGER,
                blur_score REAL
            )
            """
        )
        # Backfill new columns if migrating from older schema
        cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        for col, ddl in [
            ("inference_time_ms", "ALTER TABLE predictions ADD COLUMN inference_time_ms REAL"),
            ("model_version", "ALTER TABLE predictions ADD COLUMN model_version TEXT"),
            ("ground_truth_label", "ALTER TABLE predictions ADD COLUMN ground_truth_label TEXT"),
            ("generator_label", "ALTER TABLE predictions ADD COLUMN generator_label TEXT"),
            ("generator_json", "ALTER TABLE predictions ADD COLUMN generator_json TEXT"),
            ("file_size", "ALTER TABLE predictions ADD COLUMN file_size INTEGER"),
            ("blur_score", "ALTER TABLE predictions ADD COLUMN blur_score REAL"),
        ]:
            if col not in cols:
                conn.execute(ddl)
        conn.commit()

def _to_2d_feature(f: torch.Tensor) -> torch.Tensor:
    if f is None:
        raise RuntimeError("Empty feature tensor")
    if f.dim() == 4:
        # [N, C, H, W] or [N, H, W, C] -> mean pool spatial
        if f.shape[1] in (256, 384, 512, 768, 1024):
            return f.mean(dim=(2, 3))
        return f.permute(0, 3, 1, 2).mean(dim=(2, 3))
    if f.dim() == 3:
        return f.mean(dim=1)
    if f.dim() == 2:
        return f
    raise RuntimeError(f"Unsupported feature shape {tuple(f.shape)}; expected 2D/3D/4D")


def _build_gen_prototypes(max_per_gen: int = 80):
    global _gen_proto
    try:
        from PIL import Image
        from src.preprocess_images import get_transforms
        data_root = Path("data/standardized_jpg")
        if not data_root.exists():
            return
        # Prepare transform
        _, val_tf = get_transforms(224)
        protos: dict[str, torch.Tensor] = {}
        for label in GENERATOR_LABELS:
            paths = []
            for split in ("train", "val"):
                ai_dir = data_root / label / split / "ai"
                if ai_dir.exists():
                    paths.extend([p for p in ai_dir.glob("**/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
            if not paths:
                continue
            # limit
            paths = paths[:max_per_gen]
            feats = []
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    x = val_tf(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        _ = model(x)
                    f = _feat_buf.get("feat")
                    if f is None:
                        continue
                    f2 = _to_2d_feature(f)[0]
                    feats.append(f2)
                except Exception:
                    continue
            if feats:
                F = torch.stack(feats, dim=0)
                # L2-normalize and average then re-normalize for cosine comparisons
                F = torch.nn.functional.normalize(F, dim=1)
                m = F.mean(dim=0)
                m = torch.nn.functional.normalize(m, dim=0)
                protos[label] = m.to(DEVICE)
        _gen_proto = protos if protos else None
    except Exception:
        _gen_proto = None


@app.on_event("startup")
def startup_event():
    global model, val_transform, MODEL_VERSION, _TEMPERATURE
    init_db()
    model = load_model(MODEL_PATH, MODEL_NAME)
    # get_transforms returns (train_tfms, val_tfms)
    _, val_transform = get_transforms(image_size=224)
    try:
        if TEMPERATURE_PATH.exists():
            with open(TEMPERATURE_PATH, "r") as f:
                _TEMPERATURE = float(f.read().strip())
    except Exception:
        _TEMPERATURE = None
    # Compute model version metadata
    checksum = ""
    try:
        h = hashlib.sha256()
        with open(MODEL_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        checksum = h.hexdigest()[:12]
    except Exception:
        checksum = "unknown"
    head_type = "seq_dropout" if any(k.startswith("fc.1.") for k in model.state_dict().keys()) else "linear"
    MODEL_VERSION = {
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "checksum": checksum,
        "head": head_type,
    }

    # Register hook to capture penultimate features (input to final linear head)
    try:
        def _lin_input_hook(module, inputs, output):
            if inputs and isinstance(inputs[0], torch.Tensor):
                _feat_buf["feat"] = inputs[0].detach().to(DEVICE)

        hooked = False
        # 1) ResNet-style: model.fc (Linear) or Sequential with last Linear
        if hasattr(model, "fc") and model.fc is not None:
            target = model.fc
            if isinstance(target, torch.nn.Sequential) and len(target) > 0:
                # try last submodule if it's Linear
                last = list(target.modules())[-1]
                # Fallback to target if not linear
                (last if isinstance(last, torch.nn.Linear) else target).register_forward_hook(_lin_input_hook)
            else:
                target.register_forward_hook(_lin_input_hook)
            hooked = True

        # 2) ViT/Swin-style: model.head (Linear) or Sequential
        if not hooked and hasattr(model, "head") and model.head is not None:
            target = model.head
            if isinstance(target, torch.nn.Sequential) and len(target) > 0:
                last = list(target.modules())[-1]
                (last if isinstance(last, torch.nn.Linear) else target).register_forward_hook(_lin_input_hook)
            else:
                target.register_forward_hook(_lin_input_hook)
            hooked = True

        # 3) Fallback: find last Linear module in the model and hook it
        if not hooked:
            linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
            if linear_layers:
                linear_layers[-1].register_forward_hook(_lin_input_hook)
                hooked = True
    except Exception:
        pass

    # Build generator prototypes if requested and when no head is configured
    try:
        if not GENERATOR_HEAD_PATH:
            proto_mode = os.environ.get("GENERATOR_PROTOTYPE_MODE", "0")
            if proto_mode in ("1", "true", "yes"):
                _build_gen_prototypes(max_per_gen=int(os.environ.get("GEN_HEAD_MAX_PER_GEN", "80")))
    except Exception:
        pass


def read_imagefile(file) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def preprocess_for_model(pil_img: Image.Image, transform) -> torch.Tensor:
    return transform(pil_img).unsqueeze(0)  # add batch dim


def _ensure_gen_head_loaded():
    global _gen_head
    if _gen_head is None and GENERATOR_HEAD_PATH:
        state = torch.load(GENERATOR_HEAD_PATH, map_location=DEVICE)
        weight = state.get("weight")
        bias = state.get("bias")
        if weight is not None and bias is not None:
            n_classes, dim = weight.shape
            lin = torch.nn.Linear(dim, n_classes, bias=True).to(DEVICE)
            with torch.no_grad():
                lin.weight.copy_(weight)
                lin.bias.copy_(bias)
            _gen_head = lin.eval()


def _predict_generator_from_feat(feat: torch.Tensor):
    try:
        _ensure_gen_head_loaded()
        feat2 = _to_2d_feature(feat)
        # Prefer trained head if available and compatible
        if _gen_head is not None:
            try:
                in_features = getattr(_gen_head, 'in_features', None)
                if in_features is not None and feat2.shape[1] != in_features:
                    print(
                        f"[generator] Dimension mismatch: head expects {in_features}, got feature dim {feat2.shape[1]}. "
                        f"Falling back to prototype-based generator likelihoods if available.")
                else:
                    glogits = _gen_head(feat2)
                    gprobs = torch.softmax(glogits, dim=1)[0].detach().cpu().tolist()
                    labels = GENERATOR_LABELS[: len(gprobs)]
                    return {labels[i]: float(gprobs[i]) for i in range(len(gprobs))}
            except Exception:
                pass
        # Fallback: prototype-based cosine similarity -> softmax
        if _gen_proto:
            with torch.no_grad():
                v = torch.nn.functional.normalize(feat2[0], dim=0)
                sims = []
                labs = []
                for lab in GENERATOR_LABELS:
                    proto = _gen_proto.get(lab)
                    if proto is None:
                        continue
                    s = torch.dot(v, proto)
                    sims.append(s)
                    labs.append(lab)
                if sims:
                    sims_t = torch.stack(sims)
                    # temperature for softmax over sims
                    temp = float(os.environ.get("GENERATOR_PROTO_TEMPERATURE", "0.1"))
                    scores = torch.softmax(sims_t / max(1e-6, temp), dim=0).cpu().tolist()
                    return {labs[i]: float(scores[i]) for i in range(len(labs))}
        return None
    except Exception:
        return None


def _infer_generator_from_image(img: Image.Image) -> dict | None:
    try:
        tensor = preprocess_for_model(img, val_transform).to(DEVICE)
        with torch.no_grad():
            _ = model(tensor)  # triggers hook to populate _feat_buf["feat"]
        feat = _feat_buf.get("feat")
        if feat is None:
            return None
        return _predict_generator_from_feat(feat)
    except Exception:
        return None


@app.post("/predict")
async def predict(file: UploadFile = File(...), tta: int = 0):
    # Basic file type check
    content = await file.read()
    mime = mimetypes.guess_type(file.filename)[0]
    if mime is None or not mime.startswith("image"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    img = read_imagefile(content)
    tensor = preprocess_for_model(img, val_transform).to(DEVICE)

    start_t = time.perf_counter()
    with torch.no_grad():
        if tta:
            # Simple TTA: average original and horizontal flip
            t1 = tensor
            t2 = torch.flip(tensor, dims=[-1])
            o1 = model(t1)
            o2 = model(t2)
            outputs = (o1 + o2) / 2.0
        else:
            outputs = model(tensor)
        if _TEMPERATURE and _TEMPERATURE > 0:
            outputs = outputs / _TEMPERATURE
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prob_ai = float(probs[0, 0].cpu().item())      # assuming class order [ai, nature]
        prob_nature = float(probs[0, 1].cpu().item())
        # Compute a simple blur metric early to allow adaptive thresholding
        try:
            edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
            blur_score_early = float(sum(edges.getdata())) / (edges.width * edges.height * 255.0)
        except Exception:
            blur_score_early = None

        # Effective threshold for AI decision
        eff_th = THRESHOLD_AI
        if THRESHOLD_AI_ADAPT_K > 0.0 and blur_score_early is not None:
            # If image is very crisp (low blur), raise threshold a bit
            eff_th = THRESHOLD_AI + THRESHOLD_AI_ADAPT_K * max(0.0, THRESHOLD_AI_ADAPT_FLOOR - blur_score_early)
        eff_th = float(min(max(eff_th, 0.0), 0.99))

        pred_label = "ai" if prob_ai >= eff_th else "nature"

        # Generator head: compute multi-way probabilities from captured features
        gen_json = None
        try:
            feat = _feat_buf.get("feat")
            if feat is not None:
                gen = _predict_generator_from_feat(feat)
                if isinstance(gen, dict) and len(gen) > 0:
                    gen_json = gen
        except Exception:
            gen_json = None
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    # Persist uploaded image
    ext = Path(file.filename).suffix.lower() or ".jpg"
    uid = uuid.uuid4().hex
    stored_name = f"{uid}{ext}"
    stored_path = UPLOAD_DIR / stored_name
    try:
        with open(stored_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded image: {e}")

    # Quality signals
    file_size = len(content)
    try:
        # Simple blur metric (recomputed if needed)
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        blur_score = float(sum(edges.getdata())) / (edges.width * edges.height * 255.0)
    except Exception:
        blur_score = blur_score_early if 'blur_score_early' in locals() else None

    # Record to DB
    confidence = max(prob_ai, prob_nature)
    width, height = img.size
    now = datetime.now(timezone.utc).isoformat()
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at, filename, stored_path, mime_type, width, height,
                prediction, prob_ai, prob_nature, confidence, source,
                inference_time_ms, model_version, ground_truth_label, generator_label, generator_json,
                file_size, blur_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                file.filename,
                stored_name,
                mime or "image/unknown",
                width,
                height,
                pred_label,
                prob_ai,
                prob_nature,
                confidence,
                "upload",
                float(elapsed_ms),
                json.dumps(MODEL_VERSION or {}),
                None,
                None,
                json.dumps(gen_json) if gen_json is not None else None,
                int(file_size),
                float(blur_score) if blur_score is not None else None,
            ),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    image_url = f"/images/{stored_name}"
    return JSONResponse({
        "id": row_id,
        "prediction": pred_label,
        "probabilities": {"ai": prob_ai, "nature": prob_nature},
        "confidence": confidence,
        "filename": file.filename,
        "image_url": image_url,
        "created_at": now,
        "model_version": MODEL_VERSION,
        "inference_time_ms": elapsed_ms,
        "mime_type": mime or "image/unknown",
        "width": width,
        "height": height,
        "file_size": file_size,
        "blur_score": blur_score,
        "generator_json": gen_json,
    })


@app.post("/maintenance/backfill_generators")
def backfill_generators(limit: int = 200):
    """
    Recompute and fill generator_json for rows missing it by running inference
    on the stored images. Returns number of rows updated.
    """
    if not GENERATOR_HEAD_PATH:
        raise HTTPException(status_code=400, detail="GENERATOR_HEAD_PATH not configured on server.")
    updated = 0
    with get_db_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, stored_path FROM predictions
            WHERE (generator_json IS NULL OR generator_json = '')
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        for r in rows:
            p = UPLOAD_DIR / r["stored_path"]
            try:
                if not p.exists():
                    continue
                img = Image.open(p).convert("RGB")
                gen = _infer_generator_from_image(img)
                if gen is not None:
                    conn.execute(
                        "UPDATE predictions SET generator_json = ? WHERE id = ?",
                        (json.dumps(gen), int(r["id"]))
                    )
                    updated += 1
            except Exception:
                continue
        conn.commit()
    return {"updated": updated}


class FeedbackPayload(BaseModel):
    ground_truth_label: str | None = None  # "ai" | "nature"
    generator_label: str | None = None     # e.g., "midjourney"


@app.post("/feedback/{pid}")
def post_feedback(pid: int, payload: FeedbackPayload):
    with get_db_conn() as conn:
        cur = conn.execute(
            "UPDATE predictions SET ground_truth_label = ?, generator_label = ? WHERE id = ?",
            (payload.ground_truth_label, payload.generator_label, pid),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Not found")
    return {"status": "ok", "id": pid}


@app.get("/stats")
def get_stats():
    with get_db_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        ai_count = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE prediction = ?",
            ("ai",),
        ).fetchone()["c"]
        avg_conf_row = conn.execute(
            "SELECT AVG(confidence) AS a FROM predictions"
        ).fetchone()
        avg_conf = float(avg_conf_row["a"]) if avg_conf_row["a"] is not None else 0.0

        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        recent = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE created_at >= ?",
            (since,),
        ).fetchone()["c"]

        # Surrogate accuracy over time: daily average confidence for last 14 days
        series = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, AVG(confidence) AS avg_conf
            FROM predictions
            GROUP BY day
            ORDER BY day ASC
            LIMIT 14
            """
        ).fetchall()
        accuracy_over_time = [
            {"date": row["day"], "accuracy": round(float(row["avg_conf"]) * 100, 2)}
            for row in series
        ]

        # Distribution of predictions
        dist_rows = conn.execute(
            "SELECT prediction, COUNT(*) AS c FROM predictions GROUP BY prediction"
        ).fetchall()
        distribution = {
            row["prediction"]: row["c"] for row in dist_rows
        }

    ai_rate = (ai_count / total) if total else 0.0
    return {
        "total": total,
        "ai_rate": ai_rate,
        "avg_confidence": avg_conf,
        "recent_activity": recent,
        "accuracy_over_time": accuracy_over_time,
        "distribution": distribution,
    }


@app.get("/stats/advanced")
def get_stats_advanced():
    with get_db_conn() as conn:
        # Moving average of avg_confidence (last 30 days)
        series = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, AVG(confidence) AS avg_conf
            FROM predictions
            GROUP BY day
            ORDER BY day ASC
            LIMIT 30
            """
        ).fetchall()
        daily = [
            {"date": row["day"], "avg_conf": float(row["avg_conf"]) if row["avg_conf"] is not None else 0.0}
            for row in series
        ]

        # Top uncertain (lowest confidence)
        top_uncertain = conn.execute(
            """
            SELECT id, filename, stored_path, confidence
            FROM predictions
            ORDER BY confidence ASC
            LIMIT 10
            """
        ).fetchall()
        uncertain_items = [
            {
                "id": r["id"],
                "filename": r["filename"],
                "image_url": f"/images/{r['stored_path']}",
                "confidence": float(r["confidence"]) if r["confidence"] is not None else 0.0,
            }
            for r in top_uncertain
        ]

        # Per-source distribution
        src_rows = conn.execute(
            "SELECT source, COUNT(*) AS c FROM predictions GROUP BY source"
        ).fetchall()
        per_source = { (row["source"] or "upload"): row["c"] for row in src_rows }

    return {"daily_confidence": daily, "top_uncertain": uncertain_items, "per_source": per_source}


@app.get("/export.csv")
def export_csv(
    label: str | None = Query(None, pattern="^(ai|real)$"),
    date_from: str | None = None,  # ISO date (YYYY-MM-DD)
    date_to: str | None = None,
):
    where = []
    params: list[object] = []
    if label == "ai":
        where.append("prediction = 'ai'")
    elif label == "real":
        where.append("prediction = 'nature'")
    if date_from:
        where.append("substr(created_at,1,10) >= ?")
        params.append(date_from)
    if date_to:
        where.append("substr(created_at,1,10) <= ?")
        params.append(date_to)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    with get_db_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT id, created_at, filename, stored_path, mime_type, width, height,
                   prediction, prob_ai, prob_nature, confidence, source,
                   inference_time_ms, model_version, ground_truth_label, generator_label,
                   file_size, blur_score
            FROM predictions
            {where_sql}
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()

    # Build streaming CSV response
    import csv
    from io import StringIO

    def _iter_csv():
        header = [
            "id","created_at","filename","stored_path","mime_type","width","height",
            "prediction","prob_ai","prob_nature","confidence","source",
            "inference_time_ms","model_version","ground_truth_label","generator_label",
            "file_size","blur_score"
        ]
        sio = StringIO()
        writer = csv.writer(sio)
        writer.writerow(header)
        yield sio.getvalue()
        sio.seek(0); sio.truncate(0)
        for r in rows:
            writer.writerow([
                r["id"], r["created_at"], r["filename"], r["stored_path"], r["mime_type"], r["width"], r["height"],
                r["prediction"], r["prob_ai"], r["prob_nature"], r["confidence"], r["source"],
                r["inference_time_ms"], r["model_version"], r["ground_truth_label"], r["generator_label"],
                r["file_size"], r["blur_score"],
            ])
            yield sio.getvalue()
            sio.seek(0); sio.truncate(0)

    headers = {"Content-Disposition": "attachment; filename=export.csv"}
    return StreamingResponse(_iter_csv(), media_type="text/csv", headers=headers)


@app.get("/predictions")
def list_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=200),
    label: str | None = Query(None, pattern="^(ai|real)$"),
    q: str | None = None,
    sort: str = Query("newest", pattern="^(newest|oldest)$"),
):
    offset = (page - 1) * page_size
    where = []
    params: list[object] = []
    if label == "ai":
        where.append("prediction = 'ai'")
    elif label == "real":
        where.append("prediction = 'nature'")
    if q:
        where.append("filename LIKE ?")
        params.append(f"%{q}%")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    order_sql = " ORDER BY created_at DESC" if sort == "newest" else " ORDER BY created_at ASC"

    with get_db_conn() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS c FROM predictions{where_sql}", params).fetchone()["c"]
        rows = conn.execute(
            f"""
            SELECT id, created_at, filename, stored_path, prediction, confidence, source
            FROM predictions
            {where_sql}
            {order_sql}
            LIMIT ? OFFSET ?
            """,
            (*params, page_size, offset),
        ).fetchall()

    items = [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "filename": r["filename"],
            "image_url": f"/images/{r['stored_path']}",
            "label": "AI" if r["prediction"] == "ai" else "Real",
            "confidence": round(float(r["confidence"]) * 100, 2),
            "source": r["source"] or "upload",
        }
        for r in rows
    ]
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@app.get("/predictions/{pid}")
def get_prediction(pid: int):
    with get_db_conn() as conn:
        r = conn.execute(
            """
            SELECT id, created_at, filename, stored_path, mime_type, width, height,
                   prediction, prob_ai, prob_nature, confidence, source,
                   inference_time_ms, model_version, ground_truth_label, generator_label,
                   generator_json, file_size, blur_score
            FROM predictions WHERE id = ?
            """,
            (pid,),
        ).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "id": r["id"],
        "created_at": r["created_at"],
        "filename": r["filename"],
        "image_url": f"/images/{r['stored_path']}",
        "mime_type": r["mime_type"],
        "width": r["width"],
        "height": r["height"],
        "prediction": r["prediction"],
        "probabilities": {"ai": r["prob_ai"], "nature": r["prob_nature"]},
        "confidence": r["confidence"],
        "source": r["source"] or "upload",
        "inference_time_ms": r["inference_time_ms"],
        "model_version": json.loads(r["model_version"]) if r["model_version"] else None,
        "ground_truth_label": r["ground_truth_label"],
        "generator_label": r["generator_label"],
        "generator_json": json.loads(r["generator_json"]) if r["generator_json"] else None,
        "file_size": r["file_size"],
        "blur_score": r["blur_score"],
    }


@app.post("/admin/backfill-generators")
def backfill_generators(limit: int = Query(200, ge=1, le=2000), x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")):
    # Simple admin guard via header token if ADMIN_TOKEN is set
    admin_token = os.environ.get("ADMIN_TOKEN")
    if admin_token:
        if not x_admin_token or x_admin_token != admin_token:
            raise HTTPException(status_code=401, detail="Unauthorized")

    _ensure_gen_head_loaded()
    if _gen_head is None:
        raise HTTPException(status_code=400, detail="Generator head not configured. Set GENERATOR_HEAD_PATH.")

    with get_db_conn() as conn:
        rows = conn.execute(
            "SELECT id, stored_path FROM predictions WHERE generator_json IS NULL LIMIT ?",
            (limit,),
        ).fetchall()

    updated = 0
    missing: list[int] = []
    errors: list[int] = []
    for r in rows:
        pid = r["id"]
        img_path = UPLOAD_DIR / r["stored_path"]
        if not img_path.exists():
            missing.append(pid)
            continue
        try:
            with open(img_path, "rb") as f:
                content = f.read()
            pil = read_imagefile(content)
            tensor = preprocess_for_model(pil, val_transform).to(DEVICE)
            with torch.no_grad():
                _ = model(tensor)  # populate _feat_buf via hook
                feat = _feat_buf.get("feat")
                if feat is None:
                    errors.append(pid)
                    continue
                gen_json = _predict_generator_from_feat(feat)
                if gen_json is None:
                    errors.append(pid)
                    continue
            with get_db_conn() as conn:
                conn.execute(
                    "UPDATE predictions SET generator_json = ? WHERE id = ?",
                    (json.dumps(gen_json), pid),
                )
                conn.commit()
            updated += 1
        except Exception:
            errors.append(pid)

    return {
        "status": "ok",
        "selected": len(rows),
        "updated": updated,
        "missing": missing,
        "errors": errors,
    }


# Optional: simple health endpoint
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)