"""
AIlobotomy — операционная для локальных языковых моделей.

Запуск:  python server.py   (или start.bat на Windows)
Интерфейс: http://127.0.0.1:7861

В памяти живут две копии модели: «донор» (всегда здоровый) и «пациент».
Все операции записываются в журнал. Журнал можно редактировать: выключать
и удалять отдельные операции — пациент пересобирается из донора заново.
"""

import copy
import gc
import json
import math
import os
import random
import re
import threading
import time
import webbrowser
from pathlib import Path

import torch
from torch import nn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
SAVED_DIR = APP_DIR / "saved"
for d in (MODELS_DIR, SAVED_DIR):
    d.mkdir(exist_ok=True)

PRESETS = [
    {"id": "HuggingFaceTB/SmolLM2-135M-Instruct", "size": "135M", "note": "Крошка, мгновенно грузится. По-русски почти не говорит"},
    {"id": "HuggingFaceTB/SmolLM2-360M-Instruct", "size": "360M", "note": "Маленькая и быстрая, лучше отвечает по-английски"},
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "size": "0.5B", "note": "Лучший старт: быстрая, понимает русский, 24 слоя"},
    {"id": "Qwen/Qwen3-0.6B", "size": "0.6B", "note": "Новее и умнее, 28 слоёв"},
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "size": "1.1B", "note": "Классическая архитектура Llama, 22 слоя"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "size": "1.5B", "note": "Заметно умнее, нужна видеокарта или терпение"},
    {"id": "Qwen/Qwen3-1.7B", "size": "1.7B", "note": "Хороший русский, 28 слоёв"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "size": "3B", "note": "Для видеокарт от 8 ГБ"},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "size": "1B", "note": "Требует принять лицензию на Hugging Face и войти"},
]

IQ_TEST = [
    ("Какая столица Франции? Ответь одним словом.", r"париж|paris"),
    ("Сколько будет 2 + 2? Ответь числом.", r"(?<!\d)4(?!\d)|четыр"),
    ("Какого цвета ясное небо днём? Ответь одним словом.", r"голуб|син|blue"),
    ("Сколько дней в неделе? Ответь числом.", r"(?<!\d)7(?!\d)|сем"),
    ("Кто написал роман «Война и мир»?", r"толст|tolstoy"),
    ("Какая самая большая планета Солнечной системы?", r"юпитер|jupiter"),
    ("Сколько ног у паука? Ответь числом.", r"(?<!\d)8(?!\d)|восем"),
    ("Какая столица Японии? Ответь одним словом.", r"токио|tokyo"),
    ("Какая химическая формула воды?", r"h2o|h₂o|н2о"),
    ("Сколько будет 10 умножить на 3? Ответь числом.", r"(?<!\d)30(?!\d)|тридцат"),
]

REFERENCE_TEXT = (
    "Утром над рекой стоял густой туман. Рыбаки медленно выходили к берегу, "
    "проверяли снасти и негромко переговаривались. Когда солнце поднялось выше, "
    "туман рассеялся, и над водой закружились первые чайки. День обещал быть "
    "тёплым и спокойным, и никто никуда не торопился."
)


class OpError(Exception):
    """Ошибка до изменения весов — пересборка не нужна."""


class State:
    tok = None
    donor = None
    patient = None
    name = None
    device = "cpu"
    donor_device = "cpu"
    dtype = torch.float32
    journal = []
    map = []
    next_id = 1
    params_m = 0
    donor_tests = None
    lock = threading.RLock()
    stop = threading.Event()


S = State()


# ---------------------------------------------------------------- утилиты

def need():
    if S.patient is None:
        raise HTTPException(400, "Сначала загрузите модель")


def text_cfg(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


LAYER_PATHS = ["model.layers", "model.language_model.layers", "transformer.h",
               "gpt_neox.layers", "model.decoder.layers", "transformer.blocks"]


def find_layers(model):
    for path in LAYER_PATHS:
        obj, parent, ok = model, None, True
        for part in path.split("."):
            if hasattr(obj, part):
                parent, obj = obj, getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList):
            return obj, parent, path.split(".")[-1]
    raise OpError("Не нашёл стек слоёв у этой архитектуры")


def final_norm(model):
    for path in ["model.norm", "model.language_model.norm", "transformer.ln_f",
                 "gpt_neox.final_layer_norm", "model.decoder.final_layer_norm", "transformer.norm_f"]:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def n_layers(model=None):
    return len(find_layers(model or S.patient)[0])


def fmt(idxs):
    idxs = sorted(set(idxs))
    if not idxs:
        return "—"
    out, start, prev = [], idxs[0], idxs[0]
    for i in idxs[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(str(start) if start == prev else f"{start}–{prev}")
        if i is not None:
            start = prev = i
    return ", ".join(out)


def parse_spec(spec, n, what="слоёв"):
    if isinstance(spec, list):
        out = [int(x) for x in spec]
    else:
        out = []
        s = str(spec).replace(" ", "")
        if s.lower() in ("all", "все", "*"):
            return list(range(n))
        for chunk in s.split(","):
            if not chunk:
                continue
            if "-" in chunk[1:]:
                a, b = chunk.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(chunk))
    bad = [i for i in out if i < 0 or i >= n]
    if bad:
        raise OpError(f"Нет таких {what}: {bad}. Доступно 0–{n - 1}")
    if not out:
        raise OpError(f"Не выбрано ни одного из {what}")
    return sorted(set(out))


def sel_layers(p):
    return parse_spec(p.get("layers", []), n_layers())


def tag(idxs, kind, text):
    for i in idxs:
        S.map[i]["tags"].append({"k": kind, "t": text})


def rebuild(idxs):
    """Пересобирает стек из старых индексов; повтор индекса = копия слоя."""
    layers, parent, attr = find_layers(S.patient)
    used, new, new_map = set(), [], []
    for i in idxs:
        dup = i in used
        new.append(copy.deepcopy(layers[i]) if dup else layers[i])
        entry = {"src": S.map[i]["src"], "tags": list(S.map[i]["tags"])}
        if dup:
            entry["tags"].append({"k": "copy", "t": "копия"})
        new_map.append(entry)
        used.add(i)
    setattr(parent, attr, nn.ModuleList(new))
    S.map = new_map
    for n, layer in enumerate(new):
        for m in layer.modules():
            if hasattr(m, "layer_idx"):
                m.layer_idx = n
    for cfg in {id(c): c for c in (S.patient.config, text_cfg(S.patient))}.values():
        for a in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, a) and isinstance(getattr(cfg, a), int):
                setattr(cfg, a, len(new))
        lt = getattr(cfg, "layer_types", None)
        if lt and len(lt) == len(layers):
            cfg.layer_types = [lt[i] for i in idxs]


def classify(name):
    n = name.lower()
    if "norm" in n or "ln_" in n or n.startswith("ln"):
        return "norm"
    if "attn" in n or "attention" in n:
        return "attn"
    if "mlp" in n or "feed_forward" in n or "ffn" in n or "expert" in n:
        return "mlp"
    return "other"


def pick_params(module, target):
    for name, p in module.named_parameters():
        c = classify(name)
        if target == "norm":
            if c == "norm":
                yield p
        elif p.dim() >= 2 and (target == "all" or c == target):
            yield p


TARGET_RU = {"all": "все веса", "attn": "внимание", "mlp": "MLP", "norm": "нормализация", "embed": "эмбеддинги"}


def sub(layer, names):
    for n in names:
        m = getattr(layer, n, None)
        if m is not None:
            return m
    return None


def attn_out(layer):
    a = sub(layer, ("self_attn", "attn", "attention", "self_attention"))
    return sub(a, ("o_proj", "c_proj", "dense", "out_proj", "wo")) if a is not None else None


def mlp_out(layer):
    m = sub(layer, ("mlp", "feed_forward", "ffn"))
    return sub(m, ("down_proj", "c_proj", "dense_4h_to_h", "fc2", "w2")) if m is not None else None


def split_out(out):
    return (out[0], out[1:]) if isinstance(out, tuple) else (out, None)


def join_out(h, rest):
    return (h,) + tuple(rest) if rest is not None else h


def untie():
    emb_in = S.patient.get_input_embeddings()
    emb_out = S.patient.get_output_embeddings()
    if emb_out is not None and emb_out.weight.data_ptr() == emb_in.weight.data_ptr():
        emb_out.weight = nn.Parameter(emb_out.weight.detach().clone())
        for cfg in (S.patient.config, text_cfg(S.patient)):
            cfg.tie_word_embeddings = False
    return emb_in, emb_out


def word_variants(w):
    vs = [w, " " + w]
    if w[:1].islower():
        vs += [w[:1].upper() + w[1:], " " + w[:1].upper() + w[1:]]
    return [S.tok.encode(v, add_special_tokens=False) for v in vs]


# ---------------------------------------------------------------- операции

def op_delete(p):
    idx = sel_layers(p)
    n = n_layers()
    keep = [i for i in range(n) if i not in idx]
    if not keep:
        raise OpError("Нельзя удалить все слои — пациенту нужен хотя бы один")
    rebuild(keep)
    return f"Удалены слои {fmt(idx)}"


def op_duplicate(p):
    idx = sel_layers(p)
    times = int(p.get("times", 1))
    if times < 1:
        raise OpError("Число копий должно быть не меньше 1")
    order = []
    for i in range(n_layers()):
        order.append(i)
        if i in idx:
            order += [i] * times
    rebuild(order)
    return f"Слои {fmt(idx)} продублированы ×{times}"


def op_shuffle(p):
    idx = sel_layers(p)
    rng = random.Random(int(p["seed"]))
    sh = idx[:]
    rng.shuffle(sh)
    order = list(range(n_layers()))
    for a, b in zip(idx, sh):
        order[a] = b
    rebuild(order)
    tag(idx, "move", "перемешан")
    return f"Перемешаны слои {fmt(idx)} (seed {p['seed']})"


def op_reverse(p):
    idx = sel_layers(p)
    order = list(range(n_layers()))
    for a, b in zip(idx, reversed(idx)):
        order[a] = b
    rebuild(order)
    tag(idx, "move", "перевёрнут")
    return f"Порядок слоёв {fmt(idx)} развёрнут"


@torch.no_grad()
def op_noise(p):
    strength = float(p.get("strength", 0.3))
    target = p.get("target", "all")
    g = torch.Generator().manual_seed(int(p["seed"]))

    def apply(par):
        x = par.data.float()
        base = x.abs().mean() if (par.dim() < 2 or x.numel() < 2) else x.std()
        noise = torch.randn(par.shape, generator=g).to(par.device)
        par.data.copy_((x + noise * base * strength).to(par.dtype))

    if target == "embed":
        apply(S.patient.get_input_embeddings().weight)
        return f"Шум {strength:g}σ в эмбеддингах"
    idx = sel_layers(p)
    layers = find_layers(S.patient)[0]
    for i in idx:
        for par in pick_params(layers[i], target):
            apply(par)
    tag(idx, "noise", f"шум {strength:g}σ · {TARGET_RU.get(target, target)}")
    return f"Шум {strength:g}σ ({TARGET_RU.get(target, target)}) в слоях {fmt(idx)}"


@torch.no_grad()
def op_scale(p):
    idx = sel_layers(p)
    f = float(p.get("factor", 1.5))
    target = p.get("target", "all")
    layers = find_layers(S.patient)[0]
    for i in idx:
        for par in pick_params(layers[i], target):
            par.data.mul_(f)
    tag(idx, "scale", f"веса ×{f:g} · {TARGET_RU.get(target, target)}")
    return f"Веса ({TARGET_RU.get(target, target)}) слоёв {fmt(idx)} умножены на {f:g}"


def op_gain(p):
    idx = sel_layers(p)
    k = float(p.get("k", 0))
    layers = find_layers(S.patient)[0]

    def hook(mod, args, kwargs, out):
        x = args[0] if args else kwargs.get("hidden_states")
        h, rest = split_out(out)
        return join_out(x + k * (h - x), rest)

    for i in idx:
        layers[i].register_forward_hook(hook, with_kwargs=True)
    tag(idx, "gain", "выключен" if k == 0 else f"громкость ×{k:g}")
    return f"Громкость слоёв {fmt(idx)} × {k:g}"


@torch.no_grad()
def op_heads(p):
    idx = sel_layers(p)
    cfg = text_cfg(S.patient)
    nh = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    hd = getattr(cfg, "head_dim", None) or hidden // nh
    heads = parse_spec(p.get("heads", "0"), nh, "голов")
    layers = find_layers(S.patient)[0]
    projs = [attn_out(layers[i]) for i in idx]
    if any(pr is None for pr in projs):
        raise OpError("Не нашёл выходную проекцию внимания у этой архитектуры")
    for pr in projs:
        conv1d = type(pr).__name__ == "Conv1D"
        for h in heads:
            if conv1d:
                pr.weight[h * hd:(h + 1) * hd, :] = 0
            else:
                pr.weight[:, h * hd:(h + 1) * hd] = 0
    tag(idx, "heads", f"головы {fmt(heads)} выкл")
    return f"Отключены головы {fmt(heads)} в слоях {fmt(idx)}"


@torch.no_grad()
def zero_proj(p, getter, kind, label):
    idx = sel_layers(p)
    layers = find_layers(S.patient)[0]
    projs = [getter(layers[i]) for i in idx]
    if any(pr is None for pr in projs):
        raise OpError("Не нашёл нужный блок у этой архитектуры")
    for pr in projs:
        pr.weight.zero_()
        if getattr(pr, "bias", None) is not None:
            pr.bias.zero_()
    tag(idx, kind, label)
    return idx


def op_attn_off(p):
    return f"Отключено внимание в слоях {fmt(zero_proj(p, attn_out, 'attn', 'без внимания'))}"


def op_mlp_off(p):
    return f"Обнулены MLP в слоях {fmt(zero_proj(p, mlp_out, 'mlp', 'без MLP'))}"


@torch.no_grad()
def op_prune(p):
    idx = sel_layers(p)
    pct = float(p.get("percent", 50))
    frac = min(max(pct / 100, 0.0), 1.0)
    target = p.get("target", "all")
    layers = find_layers(S.patient)[0]
    g = torch.Generator().manual_seed(0)
    for i in idx:
        for par in pick_params(layers[i], target):
            a = par.data.abs().float().flatten()
            if a.numel() > 2_000_000:
                a = a[torch.randint(0, a.numel(), (2_000_000,), generator=g).to(a.device)]
            thr = torch.quantile(a, frac) if frac < 1 else a.max() + 1
            par.data.masked_fill_(par.data.abs().float() < thr, 0)
    tag(idx, "prune", f"обрезка {pct:g}%")
    return f"Обрезано {pct:g}% самых слабых весов ({TARGET_RU.get(target, target)}) в слоях {fmt(idx)}"


@torch.no_grad()
def op_quant(p):
    idx = sel_layers(p)
    bits = int(p.get("bits", 3))
    target = p.get("target", "all")
    layers = find_layers(S.patient)[0]
    for i in idx:
        for par in pick_params(layers[i], target):
            x = par.data.float()
            if bits <= 1:
                q = torch.sign(x) * x.abs().mean()
            else:
                lv = 2 ** (bits - 1) - 1
                m = x.abs().max()
                s = m / lv if m > 0 else torch.tensor(1.0, device=x.device)
                q = torch.round(x / s).clamp(-lv, lv) * s
            par.data.copy_(q.to(par.dtype))
    tag(idx, "quant", f"{bits} бит")
    return f"Веса слоёв {fmt(idx)} огрублены до {bits} бит"


@torch.no_grad()
def op_swap(p):
    a, b = str(p.get("a", "")).strip(), str(p.get("b", "")).strip()
    where = p.get("where", "both")
    if not a or not b:
        raise OpError("Укажите оба слова")
    A, B = word_variants(a), word_variants(b)
    pairs, used = [], set()
    for ia, ib in zip(A, B):
        if len(ia) == len(ib) and ia != ib:
            for x, y in zip(ia, ib):
                if x != y and x not in used and y not in used:
                    pairs.append((x, y))
                    used |= {x, y}
    if not pairs:
        raise OpError(f"Слова разной длины в токенах: «{a}» = {len(A[1])}, «{b}» = {len(B[1])}. "
                      f"Попробуйте другие слова или английские аналоги")
    if where in ("input", "output"):
        emb_in, emb_out = untie()
    else:
        emb_in, emb_out = S.patient.get_input_embeddings(), S.patient.get_output_embeddings()
    tied = emb_out is not None and emb_out.weight.data_ptr() == emb_in.weight.data_ptr()

    def swap(W):
        for x, y in pairs:
            t = W[x].clone()
            W[x] = W[y]
            W[y] = t

    if where in ("input", "both"):
        swap(emb_in.weight)
    if where in ("output", "both") and emb_out is not None and not tied:
        swap(emb_out.weight)
    ru = {"input": "в восприятии", "output": "в речи", "both": "везде"}[where]
    return f"Слова «{a}» ↔ «{b}» поменяны местами {ru} ({len(pairs)} пар токенов)"


@torch.no_grad()
def op_forget(p):
    w = str(p.get("word", "")).strip()
    if not w:
        raise OpError("Укажите слово")
    ids = sorted({i for v in word_variants(w) if len(v) <= 2 for i in v})
    if not ids:
        raise OpError(f"«{w}» разбивается на слишком много кусочков — выберите слово попроще")
    emb = S.patient.get_input_embeddings().weight
    emb[ids] = 0
    return f"Стёрто слово «{w}» ({len(ids)} токенов)"


@torch.no_grad()
def mean_hidden(text, L):
    ids = S.tok(text, return_tensors="pt").to(S.device)
    hs = S.patient(**ids, output_hidden_states=True).hidden_states
    return hs[L + 1][0].float().mean(0)


def op_steer(p):
    n = n_layers()
    L = int(p.get("layer", n // 2))
    if not 0 <= L < n:
        raise OpError(f"Нет слоя {L}. Доступно 0–{n - 1}")
    pos, neg = str(p.get("pos", "")).strip(), str(p.get("neg", "")).strip()
    if not pos:
        raise OpError("Нужен хотя бы положительный текст")
    k = float(p.get("strength", 6))
    vec = mean_hidden(pos, L) - (mean_hidden(neg, L) if neg else 0)
    layer = find_layers(S.patient)[0][L]

    def hook(mod, args, kwargs, out):
        h, rest = split_out(out)
        return join_out(h + k * vec.to(h.dtype).to(h.device), rest)

    layer.register_forward_hook(hook, with_kwargs=True)
    name = p.get("name") or "эмоция"
    tag([L], "steer", f"{name} ×{k:g}")
    return f"Вживлено «{name}» в слой {L}, сила {k:g}"


OPS = {
    "delete": op_delete, "duplicate": op_duplicate, "shuffle": op_shuffle,
    "reverse": op_reverse, "noise": op_noise, "scale": op_scale, "gain": op_gain,
    "heads": op_heads, "attn_off": op_attn_off, "mlp_off": op_mlp_off,
    "prune": op_prune, "quant": op_quant, "swap": op_swap, "forget": op_forget,
    "steer": op_steer,
}


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def replay():
    S.patient = None
    free_cuda()
    S.patient = copy.deepcopy(S.donor).to(S.device).eval()
    S.map = [{"src": i, "tags": []} for i in range(n_layers(S.donor))]
    for e in S.journal:
        if not e["enabled"]:
            continue
        try:
            e["desc"] = OPS[e["type"]](e["params"])
            e["error"] = None
        except Exception as ex:
            e["error"] = str(ex)


# ---------------------------------------------------------------- состояние

def state():
    if S.patient is None:
        return {"loaded": False}
    cfg = text_cfg(S.patient)
    vram = None
    if S.device == "cuda":
        vram = {"used": torch.cuda.memory_allocated() / 2**30,
                "total": torch.cuda.get_device_properties(0).total_memory / 2**30}
    return {
        "loaded": True, "name": S.name, "device": S.device, "donor_device": S.donor_device,
        "dtype": str(S.dtype).replace("torch.", ""), "arch": type(S.patient).__name__,
        "layers": n_layers(), "layers_donor": n_layers(S.donor),
        "heads": getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None),
        "params_m": S.params_m, "map": S.map, "journal": S.journal, "vram": vram,
    }


def hf_caches():
    dirs = []
    if os.environ.get("HF_HUB_CACHE"):
        dirs.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        dirs.append(Path(os.environ["HF_HOME"]) / "hub")
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [d for d in dict.fromkeys(dirs) if d.exists()]


def dir_size(p):
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
    except OSError:
        return 0


# ---------------------------------------------------------------- API

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/models")
def models():
    found = []
    for cache in hf_caches():
        for d in cache.glob("models--*"):
            snaps = sorted((d / "snapshots").glob("*/config.json"), key=lambda f: f.stat().st_mtime)
            if not snaps:
                continue
            parts = d.name.split("--", 2)
            found.append({"id": "/".join(parts[1:]), "path": str(snaps[-1].parent),
                          "gb": round(dir_size(d / "blobs") / 2**30, 2), "where": "скачана"})
    for base, where in ((MODELS_DIR, "папка models"), (SAVED_DIR, "сохранённый пациент")):
        for d in base.iterdir():
            if (d / "config.json").exists():
                found.append({"id": d.name, "path": str(d), "gb": round(dir_size(d) / 2**30, 2), "where": where})
    have = {f["id"] for f in found}
    return {"downloaded": found, "presets": [dict(p, have=p["id"] in have) for p in PRESETS],
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@app.get("/api/state")
def get_state():
    return state()


@app.post("/api/load")
def load(req: dict = Body(...)):
    name = str(req.get("path") or req.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Укажите модель")
    dev = req.get("device", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        raise HTTPException(400, "CUDA недоступна: PyTorch установлен без поддержки видеокарты")
    dt = req.get("dtype", "auto")
    if dt == "auto":
        if dev == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif dev == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dt]
    trust = bool(req.get("trust_remote_code", False))
    with S.lock:
        S.donor = S.patient = S.tok = None
        S.name = None
        free_cuda()
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust)
            donor = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, trust_remote_code=trust)
        except Exception as e:
            raise HTTPException(400, f"Не удалось загрузить «{name}»: {e}")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        S.donor_device = "cpu" if req.get("donor_cpu") else dev
        S.device, S.dtype, S.tok = dev, dtype, tok
        S.donor = donor.to(S.donor_device).eval()
        S.name = req.get("label") or name
        S.params_m = round(sum(p.numel() for p in donor.parameters()) / 1e6)
        S.journal, S.next_id, S.donor_tests = [], 1, None
        try:
            find_layers(S.donor)
        except OpError as e:
            S.donor = None
            raise HTTPException(400, str(e))
        replay()
    return state()


@app.post("/api/unload")
def unload():
    with S.lock:
        S.donor = S.patient = S.tok = None
        free_cuda()
    return state()


@app.post("/api/op")
def add_op(req: dict = Body(...)):
    need()
    t = req.get("type")
    if t not in OPS:
        raise HTTPException(400, f"Неизвестная операция {t}")
    p = dict(req.get("params") or {})
    if t in ("shuffle", "noise"):
        s = p.get("seed")
        if s in (None, "") or int(s) < 0:
            p["seed"] = random.randint(0, 2**31 - 1)
    with S.lock:
        try:
            desc = OPS[t](p)
        except OpError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            replay()
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        S.journal.append({"id": S.next_id, "type": t, "params": p, "enabled": True, "desc": desc, "error": None})
        S.next_id += 1
    return state()


def edit_journal(fn):
    need()
    with S.lock:
        fn()
        replay()
    return state()


@app.post("/api/journal/toggle")
def toggle(req: dict = Body(...)):
    def f():
        for e in S.journal:
            if e["id"] == req.get("id"):
                e["enabled"] = not e["enabled"]
    return edit_journal(f)


@app.post("/api/journal/delete")
def delete(req: dict = Body(...)):
    return edit_journal(lambda: setattr(S, "journal", [e for e in S.journal if e["id"] != req.get("id")]))


@app.post("/api/journal/move")
def move(req: dict = Body(...)):
    def f():
        j = S.journal
        i = next((k for k, e in enumerate(j) if e["id"] == req.get("id")), None)
        d = int(req.get("dir", 0))
        if i is not None and 0 <= i + d < len(j):
            j[i], j[i + d] = j[i + d], j[i]
    return edit_journal(f)


@app.post("/api/undo")
def undo():
    return edit_journal(lambda: S.journal.pop() if S.journal else None)


@app.post("/api/reset")
def reset():
    return edit_journal(lambda: setattr(S, "journal", []))


@app.post("/api/journal/import")
def import_journal(req: dict = Body(...)):
    ops = req.get("journal") or []

    def f():
        S.journal = []
        for e in ops:
            if e.get("type") in OPS:
                S.journal.append({"id": S.next_id, "type": e["type"], "params": e.get("params", {}),
                                  "enabled": e.get("enabled", True), "desc": e.get("desc", e["type"]), "error": None})
                S.next_id += 1
    return edit_journal(f)


@app.post("/api/save")
def save(req: dict = Body(...)):
    need()
    name = re.sub(r"[^\w\-. ]", "_", str(req.get("name") or "patient")).strip() or "patient"
    out = SAVED_DIR / name
    with S.lock:
        S.patient.save_pretrained(out)
        S.tok.save_pretrained(out)
        (out / "recipe.json").write_text(json.dumps({"model": S.name, "journal": S.journal},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    hooks = [e for e in S.journal if e["enabled"] and e["type"] in ("gain", "steer")]
    note = " Громкость и эмоции живут в коде, а не в весах, поэтому в файл не попали." if hooks else ""
    return {"path": str(out), "note": note}


# ---------------------------------------------------------------- генерация

def build_inputs(prompt, system, use_chat, thinking, device):
    tok = S.tok
    if use_chat and tok.chat_template:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        except Exception:
            msgs = [{"role": "user", "content": (system + "\n\n" if system else "") + prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    text = (system + "\n\n" if system else "") + prompt
    return tok(text, return_tensors="pt").to(device)


def gen_kwargs(g):
    kw = {"max_new_tokens": int(g.get("max_new_tokens") or 200), "pad_token_id": S.tok.pad_token_id}
    t = float(g.get("temperature") or 0)
    if t > 0:
        kw.update(do_sample=True, temperature=t,
                  top_p=float(g["top_p"]) if g.get("top_p") not in (None, "") else None,
                  top_k=int(g["top_k"]) if g.get("top_k") not in (None, "") else None)
        if float(g.get("min_p") or 0) > 0:
            kw["min_p"] = float(g["min_p"])
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    if float(g.get("repetition_penalty") or 1) != 1:
        kw["repetition_penalty"] = float(g["repetition_penalty"])
    if int(g.get("no_repeat_ngram_size") or 0) > 0:
        kw["no_repeat_ngram_size"] = int(g["no_repeat_ngram_size"])
    return kw


class Stopper(StoppingCriteria):
    def __call__(self, input_ids, scores, **kw):
        return S.stop.is_set()


def seed_all(g):
    s = int(g.get("seed") if g.get("seed") not in (None, "") else 42)
    torch.manual_seed(s if s >= 0 else random.randint(0, 2**31 - 1))


@app.post("/api/generate")
def generate(req: dict = Body(...)):
    need()
    who = req.get("who", "patient")
    g = req.get("gen") or {}
    streamer = TextIteratorStreamer(S.tok, skip_prompt=True, skip_special_tokens=True)
    err = {}
    S.stop.clear()

    def run():
        with S.lock:
            try:
                model, dev = (S.patient, S.device) if who == "patient" else (S.donor, S.donor_device)
                ids = build_inputs(req.get("prompt", ""), req.get("system", ""),
                                   g.get("chat_template", True), g.get("thinking", False), dev)
                seed_all(g)
                with torch.no_grad():
                    model.generate(**ids, **gen_kwargs(g), streamer=streamer,
                                   stopping_criteria=StoppingCriteriaList([Stopper()]))
            except Exception as e:
                err["e"] = f"{type(e).__name__}: {e}"
                streamer.end()

    threading.Thread(target=run, daemon=True).start()

    def stream():
        for t in streamer:
            yield t
        if "e" in err:
            yield f"\n\n💀 Пациент не выжил: {err['e']}"

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


@app.post("/api/stop")
def stop():
    S.stop.set()
    return {"ok": True}


@torch.no_grad()
def lens_for(model, dev, prompt, k):
    ids = S.tok(prompt, return_tensors="pt").to(dev)
    hs = model(**ids, output_hidden_states=True).hidden_states
    norm, head = final_norm(model), model.get_output_embeddings()
    rows, n = [], len(hs) - 1
    for i in range(1, n + 1):
        h = hs[i][0, -1:]
        if i < n and norm is not None:
            h = norm(h)
        pr = head(h).float()[0].softmax(-1)
        v, ix = pr.topk(k)
        rows.append([{"t": S.tok.decode([int(j)]), "p": round(float(x), 4)} for x, j in zip(v, ix)])
    return rows


@app.post("/api/lens")
def lens(req: dict = Body(...)):
    need()
    prompt = req.get("prompt") or "Столица Франции — город"
    k = int(req.get("k") or 5)
    with S.lock:
        try:
            res = {"patient": lens_for(S.patient, S.device, prompt, k)}
        except Exception as e:
            res = {"patient": None, "error": f"{type(e).__name__}: {e}"}
        res["donor"] = lens_for(S.donor, S.donor_device, prompt, k)
    return res


@torch.no_grad()
def run_tests(model, dev):
    results, score = [], 0
    g = {"max_new_tokens": 40, "temperature": 0}
    for q, rx in IQ_TEST:
        try:
            ids = build_inputs(q, "", True, False, dev)
            out = model.generate(**ids, **gen_kwargs(g))
            ans = S.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            ans = f"💀 {type(e).__name__}"
        ok = bool(re.search(rx, ans.lower()))
        score += ok
        results.append({"q": q, "a": ans[:160], "ok": ok})
    try:
        ids = S.tok(REFERENCE_TEXT, return_tensors="pt").to(dev)
        loss = float(model(**ids, labels=ids["input_ids"]).loss)
        ppl = math.exp(loss) if math.isfinite(loss) and loss < 50 else None
    except Exception:
        ppl = None
    return {"score": score, "total": len(IQ_TEST), "ppl": ppl and round(ppl, 2), "items": results}


@app.post("/api/iq")
def iq():
    need()
    with S.lock:
        if S.donor_tests is None:
            S.donor_tests = run_tests(S.donor, S.donor_device)
        return {"patient": run_tests(S.patient, S.device), "donor": S.donor_tests}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("AILOBOTOMY_PORT", 7861))
    url = f"http://127.0.0.1:{port}"
    print(f"\n  AIlobotomy запущена: {url}\n  Чтобы остановить, закройте это окно или нажмите Ctrl+C\n")
    if not os.environ.get("AILOBOTOMY_NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
