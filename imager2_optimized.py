"""
Imager-2 optimizado: fine-tuning binario IA vs real sobre ConvNeXt (PyTorch + torchvision).

Verificado contra las cards y el config del checkpoint:
  - Imagers/imager-1-pro es pytorch_model.bin en formato HuggingFace. Se convierte a claves de
    torchvision sin usar transformers, y la carga aborta si no encaja >= 98 % de los tensores.
  - Defactify: columna Label_A, 0 = real, 1 = IA. Usa splits oficiales.
  - Hemg: columna label, valor 0 = AiArtData (IA).
  - Fuentes sin verificar o inutilizables quedan deshabilitadas en DATASETS.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
T0 = time.time()  # el presupuesto se cuenta desde aquí, no desde main()

import copy
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.ops import StochasticDepth
from PIL import Image, ImageFile, ImageFilter
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CFG:
    SEED = 42
    OUT = Path("/kaggle/working/imager2")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Presupuesto: Kaggle corta a 12 h por sesión; 30 min de margen.
    TOTAL_LIMIT_H = 11.5
    FINAL_RESERVE_MIN = 40        # calibración de umbral + test con TTA + escritura

    # Checkpoint de partida
    PRO_REPO = "Imagers/imager-1-pro"
    PRO_FILE = "pytorch_model.bin"
    ARCH = "auto"                 # "auto" = deducir de las dimensiones del checkpoint

    # Preprocesado
    IMG_SIZE = 224
    RESIZE_SHORT = 0              # 0 = resolución nativa; 256 = mismo preprocesado que usa el Pro

    # Optimización
    BATCH_SIZE = 32
    LR_TOP = 1e-5                 # backbone, capa más alta
    LLRD = 0.8                    # layer-wise decay: cada capa inferior se multiplica por 0.8
    LR_HEAD = 1e-4
    WEIGHT_DECAY = 0.05
    WARMUP_STEPS = 500
    GRAD_CLIP = 1.0
    LABEL_SMOOTH = 0.03
    DROP_PATH = 0.1               # = drop_path_rate del config.json del Pro
    HEAD_DROPOUT = 0.25
    EMA_DECAY = 0.9995

    # Aumentos (solo train)
    P_JPEG = 0.25
    JPEG_Q = (65, 95)
    P_BLUR = 0.10
    BLUR_SIGMA = (0.1, 1.0)

    # Evaluación y control
    EVAL_EVERY_SEC = 2400
    VAL_EVAL_MAX = 15000
    PATIENCE_EVALS = 5
    TTA_FINAL = True              # volteo horizontal en validación final y test
    RESUME = True
    NUM_WORKERS = min(4, os.cpu_count() or 2)
    PREFETCH = 4
    LOG_EVERY = 200
    MAX_NONFINITE = 300


# role: "train" (por defecto) o "external" (solo evaluación).
# split_mode: "random" = 80/10/10 por fuente; "official" = splits de la card.
DATASETS = [
    dict(name="hemg", path="Hemg/AI-Generated-vs-Real-Images-Datasets", enabled=True,
         image_col="image", label_col="label", ai_value=0, expect_name="ai",
         split_mode="random"),
    dict(name="defactify", path="Rajarshi-Roy-research/Defactify_Image_Dataset", enabled=True,
         image_col="Image", label_col="Label_A", ai_value=1,
         split_mode="official", official={"train": "train", "val": "validation", "test": "test"}),
    # Sin verificar: la card devolvió 401. Activar solo tras confirmar el valor IA
    # (ds.features["label"].names o la card).
    dict(name="dima806", path="Dima806/ai_vs_real_image_detection", enabled=False,
         image_col="image", label_col="label", ai_value=1, split_mode="random"),
    dict(name="artem9k", path="artem9k/ai-vs-human-generated-dataset", enabled=False,
         image_col="image", label_col="label", ai_value=1, split_mode="random"),
    # InfImagine/FakeImageDataset: excluido. No tiene columna de etiqueta (están en MetaData/*.csv),
    # el visor falla por un tar truncado y ImageData solo contiene imágenes generadas.
]
# Evaluación externa, no entrena. Ejemplo:
# dict(name="x", path="...", role="external", split="test", image_col="image", label_col="label", ai_value=1)
EXTRA_TEST = []


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f} min]", *a, flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# DATOS Y AUMENTOS
# ============================================================
class EnsureMinSide:
    """Sube solo las imágenes con lado menor que IMG_SIZE; el resto conserva su resolución."""
    def __init__(self, m):
        self.m = m

    def __call__(self, img):
        w, h = img.size
        if min(w, h) >= self.m:
            return img
        s = self.m / min(w, h)
        return img.resize((math.ceil(w * s), math.ceil(h * s)), Image.Resampling.BICUBIC)


class RandomJPEG:
    """Recomprime en JPEG con calidad aleatoria. Evita que el formato del fichero sea un atajo."""
    def __init__(self, p, qlo, qhi):
        self.p, self.qlo, self.qhi = p, qlo, qhi

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(self.qlo, self.qhi))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlur:
    def __init__(self, p, slo, shi):
        self.p, self.slo, self.shi = p, slo, shi

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(self.slo, self.shi)))


def make_transforms():
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    pre = [EnsureMinSide(CFG.IMG_SIZE)]
    if CFG.RESIZE_SHORT:
        pre.append(transforms.Resize(CFG.RESIZE_SHORT, interpolation=transforms.InterpolationMode.BICUBIC))
    tr = transforms.Compose(pre + [
        RandomJPEG(CFG.P_JPEG, *CFG.JPEG_Q),
        RandomBlur(CFG.P_BLUR, *CFG.BLUR_SIGMA),
        transforms.RandomCrop(CFG.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    ev = transforms.Compose(pre + [
        transforms.CenterCrop(CFG.IMG_SIZE), transforms.ToTensor(), norm,
    ])
    return tr, ev


class ImageDS(Dataset):
    """Devuelve (x, y, id_fuente), o None si la imagen no se puede decodificar (se filtra en collate)."""
    def __init__(self, hf_ds, image_col, labels, src, tf):
        self.ds, self.col, self.labels, self.src, self.tf = hf_ds, image_col, labels, src, tf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        try:
            img = self.ds[int(i)][self.col]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))
            x = self.tf(img.convert("RGB"))
        except Exception:
            return None
        return x, torch.tensor(float(self.labels[i])), self.src


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys, ss = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.as_tensor(ss, dtype=torch.long)


def loader_kwargs(persistent=True):
    nw = CFG.NUM_WORKERS
    return dict(num_workers=nw, pin_memory=(CFG.DEVICE == "cuda"), collate_fn=collate,
                persistent_workers=(persistent and nw > 0),
                prefetch_factor=(CFG.PREFETCH if nw > 0 else None))


def make_eval_loader(ds):
    return DataLoader(ds, batch_size=CFG.BATCH_SIZE, shuffle=False, **loader_kwargs(persistent=False))


def binary_labels(ds, spec):
    names = getattr(ds.features[spec["label_col"]], "names", None)
    if names:
        log(f"[{spec['name']}] nombres de clase: {list(names)}")
        if spec.get("expect_name") and spec["expect_name"] not in str(names[spec["ai_value"]]).lower():
            raise ValueError(f"ai_value={spec['ai_value']} es '{names[spec['ai_value']]}', "
                             f"no contiene '{spec['expect_name']}'")
    raw = np.asarray(ds[spec["label_col"]])
    vals = np.unique(raw).tolist()
    if len(vals) != 2 or spec["ai_value"] not in vals:
        raise ValueError(f"etiquetas {vals} no encajan con ai_value={spec['ai_value']}")
    log(f"[{spec['name']}] recuento bruto:", {v: int((raw == v).sum()) for v in vals},
        f"-> IA = valor {spec['ai_value']}")
    return (raw == spec["ai_value"]).astype(np.uint8)


def build_sources():
    rng = np.random.default_rng(CFG.SEED)
    tf_tr, tf_ev = make_transforms()
    buckets = {"train": [], "val": [], "test": [], "external": []}
    names = []
    for spec in [s for s in DATASETS if s.get("enabled")] + EXTRA_TEST:
        src = len(names)
        local = {"train": [], "val": [], "test": [], "external": []}
        try:
            col = spec["image_col"]
            if spec.get("role") == "external":
                ds = load_dataset(spec["path"], split=spec.get("split", "train"))
                y = binary_labels(ds, spec)
                local["external"].append(ImageDS(ds.select_columns([col]), col, y, src, tf_ev))
            elif spec.get("split_mode") == "official":
                for role, split in spec["official"].items():
                    ds = load_dataset(spec["path"], split=split)
                    y = binary_labels(ds, spec)
                    tf = tf_tr if role == "train" else tf_ev
                    local[role].append(ImageDS(ds.select_columns([col]), col, y, src, tf))
            else:
                ds = load_dataset(spec["path"], split="train")
                y = binary_labels(ds, spec)
                idx = rng.permutation(len(ds))
                a, b = int(0.8 * len(ds)), int(0.9 * len(ds))
                for role, ids in (("train", idx[:a]), ("val", idx[a:b]), ("test", idx[b:])):
                    sub = ds.select(ids.tolist()).select_columns([col])
                    tf = tf_tr if role == "train" else tf_ev
                    local[role].append(ImageDS(sub, col, y[ids], src, tf))
            for r in local:
                buckets[r] += local[r]
            names.append(spec["name"])
            log(f"[OK] {spec['name']} cargado")
        except Exception as e:
            log(f"[ERROR] {spec['name']} omitido -> {type(e).__name__}: {e}")
    return buckets, names


def make_sampler(parts, names):
    """Peso por grupo (fuente, clase) proporcional a sqrt(n): compensa desbalanceos sin sobreentrenar fuentes pequeñas."""
    counts = {(d.src, c): int((d.labels == c).sum()) for d in parts for c in (0, 1)}
    total = sum(math.sqrt(n) for n in counts.values() if n > 0)
    chunks = []
    for d in parts:
        w = np.zeros(len(d), dtype=np.float64)
        for c in (0, 1):
            n = counts[(d.src, c)]
            if n > 0:
                w[d.labels == c] = (math.sqrt(n) / total) / n
        chunks.append(w)
    for (src, c), n in counts.items():
        if n > 0:
            log(f"[SAMPLER] {names[src]} {'IA' if c else 'real'}: n={n:,} masa={math.sqrt(n) / total:.3f}")
    w = np.concatenate(chunks)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True)


def audit(d, name, n=300):
    """Fracción de imágenes pequeñas por clase: si difiere mucho, la resolución es un atajo."""
    rng = np.random.default_rng(0)
    ids = rng.choice(len(d), size=min(n, len(d)), replace=False)
    small = {0: [0, 0], 1: [0, 0]}
    for i in ids:
        try:
            w, h = d.ds[int(i)][d.col].size
        except Exception:
            continue
        c = int(d.labels[i])
        small[c][0] += int(min(w, h) < CFG.IMG_SIZE)
        small[c][1] += 1
    fr = {c: (round(a / b, 3) if b else None) for c, (a, b) in small.items()}
    log(f"[AUDIT] {name}: fracción con lado < {CFG.IMG_SIZE} px -> real={fr[0]} IA={fr[1]}")


# ============================================================
# MODELO Y CHECKPOINT DEL PRO
# ============================================================
class Imager2(nn.Module):
    def __init__(self, arch):
        super().__init__()
        ctor = models.convnext_base if arch == "convnext_base" else models.convnext_small
        self.backbone = ctor(weights=None)
        for m in self.backbone.modules():
            if isinstance(m, StochasticDepth):
                m.p = CFG.DROP_PATH
        dim = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0],          # LayerNorm2d: se carga desde convnext.layernorm del Pro
            nn.Flatten(1),
            nn.Linear(dim, 512), nn.GELU(), nn.Dropout(CFG.HEAD_DROPOUT),
            nn.Linear(512, 1),
        )
        nn.init.normal_(self.backbone.classifier[-1].weight, std=0.01)
        nn.init.zeros_(self.backbone.classifier[-1].bias)

    def forward(self, x):
        return self.backbone(x)


def hf_to_tv(sd):
    """Claves de ConvNextForImageClassification (HF) -> features/classifier.0 de torchvision."""
    out = {}
    for k, v in sd.items():
        if not k.startswith("convnext."):
            continue
        k2 = k[len("convnext."):]
        if k2.startswith("embeddings.patch_embeddings."):
            out["features.0.0." + k2.split(".")[-1]] = v
        elif k2.startswith("embeddings.layernorm."):
            out["features.0.1." + k2.split(".")[-1]] = v
        elif k2 in ("layernorm.weight", "layernorm.bias"):
            out["classifier.0." + k2.split(".")[-1]] = v
        elif k2.startswith("encoder.stages."):
            p = k2.split(".")
            i, rest = int(p[2]), p[3:]
            if rest[0] == "downsampling_layer":
                out[f"features.{2 * i}.{rest[1]}.{rest[2]}"] = v
            elif rest[0] == "layers":
                j, name = int(rest[1]), rest[2]
                suf = rest[3] if len(rest) > 3 else ""
                base = f"features.{2 * i + 1}.{j}."
                if name == "dwconv":
                    out[base + f"block.0.{suf}"] = v
                elif name == "layernorm":
                    out[base + f"block.2.{suf}"] = v
                elif name == "pwconv1":
                    out[base + f"block.3.{suf}"] = v
                elif name == "pwconv2":
                    out[base + f"block.5.{suf}"] = v
                elif name == "layer_scale_parameter":
                    out[base + "layer_scale"] = v.reshape(-1, 1, 1)
    return out


def load_pro(path):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, nn.Module):
        raw = raw.state_dict()
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(raw, dict) and isinstance(raw.get(key), dict):
            raw = raw[key]
    sd = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            sd[k[7:] if k.startswith("module.") else k] = v
    is_hf = any(k.startswith("convnext.") for k in sd)
    if is_hf:
        conv = hf_to_tv(sd)
    else:
        conv = {k: v for k, v in sd.items() if k.startswith(("features.", "classifier.0."))}
    stem = sd.get("convnext.embeddings.patch_embeddings.weight", sd.get("features.0.0.weight"))
    if stem is None:
        raise RuntimeError("No encuentro el stem de ConvNeXt en el checkpoint.")
    arch = {128: "convnext_base", 96: "convnext_small"}.get(int(stem.shape[0]))
    if arch is None:
        raise RuntimeError(f"Dimensión del stem {int(stem.shape[0])}: no es Base (128) ni Small (96).")
    if CFG.ARCH not in ("auto", arch):
        raise RuntimeError(f"El checkpoint es {arch} pero CFG.ARCH={CFG.ARCH}.")

    model = Imager2(arch)
    target = model.state_dict()
    conv = {"backbone." + k: v.float() for k, v in conv.items()}
    expected = [k for k in target if k.startswith(("backbone.features.", "backbone.classifier.0."))]
    load, bad_shape, nonfin, missing = {}, [], [], []
    for k in expected:
        v = conv.get(k)
        if v is None:
            missing.append(k)
        elif tuple(v.shape) != tuple(target[k].shape):
            bad_shape.append(k)
        elif not torch.isfinite(v).all():
            nonfin.append(k)
        else:
            load[k] = v
    ratio = len(load) / max(1, len(expected))
    log(f"[PRO] formato={'HF' if is_hf else 'torchvision'} arch={arch} cargados={len(load)}/{len(expected)} "
        f"({ratio:.1%}) shape_mismatch={len(bad_shape)} no_finitos={len(nonfin)} faltan={len(missing)}")
    if ratio < 0.98:
        raise RuntimeError("Carga del Pro insuficiente, no se entrena desde cero. Revisa: "
                           + ", ".join((bad_shape + missing)[:8]))
    model.load_state_dict(load, strict=False)
    return model, {"arch": arch, "formato": "HF" if is_hf else "torchvision",
                   "cargados": len(load), "esperados": len(expected)}


class EMA:
    def __init__(self, model, decay):
        self.decay, self.n = decay, 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.ep = list(self.module.parameters())

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        torch._foreach_mul_(self.ep, d)
        torch._foreach_add_(self.ep, list(model.parameters()), alpha=1 - d)


# ============================================================
# MÉTRICAS
# ============================================================
def metrics(p, y, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    if len(y) == 0:
        return {"n": 0}
    pred = (p >= thr).astype(int)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {
        "n": int(len(y)), "thr": float(thr),
        "acc": float(accuracy_score(y, pred)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "recall_ai": float(recall_score(y, pred, zero_division=0)),
        "precision_ai": float(precision_score(y, pred, zero_division=0)),
        "auc": auc,
        "cm": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def choose_threshold(p, y):
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.01):
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


@torch.inference_mode()
def predict(net, loader, tta=False):
    net.eval()
    P, Y, S = [], [], []
    for batch in tqdm(loader, leave=False):
        if batch is None:
            continue
        x, y, s = batch
        x = x.to(CFG.DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(CFG.DEVICE == "cuda")):
            p = torch.sigmoid(net(x).float().squeeze(1))
            if tta:
                p = 0.5 * (p + torch.sigmoid(net(torch.flip(x, dims=[3])).float().squeeze(1)))
        P.append(p.cpu().numpy())
        Y.append(y.numpy().astype(int))
        S.append(s.numpy())
    if not P:
        return np.zeros(0), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(P), np.concatenate(Y), np.concatenate(S)


# ============================================================
# MAIN
# ============================================================
def main():
    seed_everything(CFG.SEED)
    OUT = CFG.OUT
    OUT.mkdir(parents=True, exist_ok=True)
    DEV = CFG.DEVICE
    amp = DEV == "cuda"
    if amp:
        torch.backends.cudnn.benchmark = True
        log("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported(), "-> AMP fp16")
    train_deadline = T0 + CFG.TOTAL_LIMIT_H * 3600 - CFG.FINAL_RESERVE_MIN * 60
    log(f"Entrenamiento hasta {(train_deadline - T0) / 3600:.2f} h desde el arranque "
        f"(fin duro {CFG.TOTAL_LIMIT_H} h)")

    # 1. Checkpoint de partida
    model, pro_info = load_pro(hf_hub_download(CFG.PRO_REPO, CFG.PRO_FILE))
    arch = pro_info["arch"]
    model = model.to(DEV).to(memory_format=torch.channels_last)

    # 2. Datos
    buckets, names = build_sources()
    if not buckets["train"]:
        raise RuntimeError("Ninguna fuente de entrenamiento se pudo cargar.")
    for d in buckets["train"]:
        audit(d, names[d.src])
    train_ds = ConcatDataset(buckets["train"])
    val_ds = ConcatDataset(buckets["val"])
    test_ds = ConcatDataset(buckets["test"])
    log(f"train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    sampler = make_sampler(buckets["train"], names)
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler,
                              drop_last=True, **loader_kwargs())
    sub = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(CFG.SEED))
    val_sub_loader = make_eval_loader(Subset(val_ds, sub[:CFG.VAL_EVAL_MAX].tolist()))

    # 3. Optimización: LLRD (features[k] -> LR_TOP * LLRD^(7-k); cabeza -> LR_HEAD), sin decay en normas/bias
    groups = {}
    for n_, p in model.named_parameters():
        lid = int(n_.split(".")[2]) if n_.startswith("backbone.features.") else 8
        decay = p.ndim > 1 and "layer_scale" not in n_
        g = groups.setdefault((lid, decay), {"params": [], "weight_decay": CFG.WEIGHT_DECAY if decay else 0.0})
        g["params"].append(p)
    for (lid, _), g in groups.items():
        g["base_lr"] = CFG.LR_HEAD if lid == 8 else CFG.LR_TOP * CFG.LLRD ** (7 - lid)
        g["lr"] = g["base_lr"]
    opt = torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ema = EMA(model, CFG.EMA_DECAY)

    state = {"step": 0, "elapsed": 0.0, "best": -1.0, "hist": [], "bad_evals": 0, "n_bad": 0}
    last_p, best_p = OUT / "last.pt", OUT / "best_ema.pt"
    if CFG.RESUME and last_p.exists():
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        ema.n = ck["ema_n"]
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        log(f"Reanudado desde {last_p} en el paso {state['step']}")

    def eval_val_sub():
        p, y, s = predict(ema.module, val_sub_loader, tta=False)
        per = {names[k]: metrics(p[s == k], y[s == k]) for k in range(len(names)) if (s == k).any()}
        aucs = [v["auc"] for v in per.values() if not math.isnan(v["auc"])]
        return (float(np.mean(aucs)) if aucs else -1.0), per

    def save_last():
        tmp = last_p.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "ema_n": ema.n,
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "state": {k: state[k] for k in ("step", "elapsed", "best", "hist", "bad_evals", "n_bad")}},
                   tmp)
        os.replace(tmp, last_p)

    # 4. Entrenamiento: el coseno va por tiempo de pared, así se adapta a la GPU real y a las evaluaciones
    model.train()
    step = state["step"]
    t_train0 = time.time() - state["elapsed"]
    next_eval = time.time() + CFG.EVAL_EVERY_SEC
    run_loss, run_n, stop = 0.0, 0, False
    while not stop and time.time() < train_deadline:
        for batch in train_loader:
            now = time.time()
            if now >= train_deadline:
                stop = True
                break
            if batch is None:
                continue
            x, y, _ = batch
            x = x.to(DEV, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(DEV, non_blocking=True)
            prog = min(1.0, max(0.0, (now - t_train0) / max(1.0, train_deadline - t_train0)))
            scale = min(1.0, (step + 1) / CFG.WARMUP_STEPS) * 0.5 * (1 + math.cos(math.pi * prog))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            target = y * (1 - CFG.LABEL_SMOOTH) + CFG.LABEL_SMOOTH / 2
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(x).float().squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                state["n_bad"] += 1
                if state["n_bad"] % 50 == 1:
                    log(f"[NONFINITE] loss no finita; batch omitido (total {state['n_bad']})")
                if state["n_bad"] > CFG.MAX_NONFINITE:
                    raise RuntimeError("Demasiadas losses no finitas: revisa AMP y el checkpoint.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1
            run_loss += loss.item()
            run_n += 1
            if step % CFG.LOG_EVERY == 0:
                lr_max = max(g["lr"] for g in opt.param_groups)
                log(f"paso {step} | loss {run_loss / run_n:.4f} | lr_max {lr_max:.2e} | "
                    f"escala {scale:.3f} | {(now - t_train0) / 60:.1f} min entrenados")
                run_loss, run_n = 0.0, 0
            if time.time() >= next_eval:
                score, per = eval_val_sub()
                aucs_txt = {k: round(v["auc"], 4) for k, v in per.items()}
                state["hist"].append({"step": step, "min": round((time.time() - T0) / 60, 1),
                                      "score": score, "auc": aucs_txt})
                if score > state["best"]:
                    state["best"], state["bad_evals"] = score, 0
                    torch.save(ema.module.state_dict(), best_p)
                    tag = "[BEST]"
                else:
                    state["bad_evals"] += 1
                    tag = "[EVAL]"
                state["step"], state["elapsed"] = step, time.time() - t_train0
                save_last()
                log(f"{tag} paso {step} | score(AUC medio por fuente)={score:.4f} | "
                    f"AUC por fuente={aucs_txt} | sin mejora={state['bad_evals']}")
                next_eval = time.time() + CFG.EVAL_EVERY_SEC
                if state["bad_evals"] >= CFG.PATIENCE_EVALS:
                    log("[STOP] paciencia agotada")
                    stop = True
                    break

    # 5. Final: última evaluación, carga del mejor EMA, umbral en validación, test
    log("Entrenamiento terminado. Calibrando umbral y evaluando test.")
    del train_loader
    score, _ = eval_val_sub()
    if score > state["best"] or not best_p.exists():
        state["best"] = max(score, state["best"])
        torch.save(ema.module.state_dict(), best_p)
    ema.module.load_state_dict(torch.load(best_p, map_location=DEV))

    pv, yv, _ = predict(ema.module, make_eval_loader(val_ds), CFG.TTA_FINAL)
    thr, f1_val = choose_threshold(pv, yv)
    pt, yt, st = predict(ema.module, make_eval_loader(test_ds), CFG.TTA_FINAL)

    ext = {}
    for d in buckets["external"]:
        pe, ye, _ = predict(ema.module, make_eval_loader(d), CFG.TTA_FINAL)
        ext[names[d.src]] = metrics(pe, ye, thr)

    counts = {}
    for role in ("train", "val", "test", "external"):
        for d in buckets[role]:
            counts.setdefault(names[d.src], {"train": 0, "val": 0, "test": 0, "external": 0})[role] += len(d)

    out = {
        "arch": arch,
        "checkpoint": pro_info,
        "sources": names,
        "counts": counts,
        "train_steps": step,
        "train_minutes": round((time.time() - t_train0) / 60, 1),
        "nonfinite_batches": state["n_bad"],
        "best_score_auc_medio_por_fuente": state["best"],
        "threshold": thr,
        "val_f1_macro_at_threshold": f1_val,
        "test_overall": metrics(pt, yt, thr),
        "test_by_source": {names[k]: metrics(pt[st == k], yt[st == k], thr)
                           for k in range(len(names)) if (st == k).any()},
        "external": ext,
        "history": state["hist"],
        "nota": "Splits aleatorios por fuente salvo Defactify (oficiales). No es un test con generadores no vistos.",
    }
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    log("TEST global:", {k: round(v, 4) for k, v in out["test_overall"].items() if isinstance(v, float)})
    for k, v in out["test_by_source"].items():
        log(f"  {k}: AUC={v['auc']:.4f} F1macro={v['f1_macro']:.4f} recall_IA={v['recall_ai']:.4f}")
    for k, v in ext.items():
        log(f"  externo {k}: AUC={v['auc']:.4f} F1macro={v['f1_macro']:.4f}")
    log("Listo. Pesos:", best_p, "| resultados:", OUT / "results.json")


if __name__ == "__main__":
    main()
