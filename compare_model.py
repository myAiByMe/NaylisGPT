#!/usr/bin/env python3
"""
compare_baselines.py — Naylis v1 — Comparaison modèles de référence
====================================================================
Teste tous les modèles de référence + Naylis (pretrain & SFT).

Modèles testés :
  HuggingFace (pretrain-only, completion style) :
    - HuggingFaceTB/cosmo-1b          (1B  — même corpus que ton pretrain)
    - HuggingFaceTB/SmolLM2-135M      (135M — même taille, référence HF)
    - EleutherAI/pythia-160m          (160M — baseline industrie)
    - EleutherAI/pythia-160m-deduped  (160M — corpus dédupliqué)

  Naylis (locaux, chargés depuis .pt) :
    - naylis_pt   → ./Model/naylis_pretrain.pt   (pretrain raw)
    - naylis_sft  → ./Model/naylis_sft.pt        (SFT instruct)

Usage :
  python compare_baselines.py                              ← tous les modèles
  python compare_baselines.py --models naylis_pt,naylis_sft
  python compare_baselines.py --models cosmo,smollm2,naylis_pt
  python compare_baselines.py --quick-test                ← 4 prompts seulement
  python compare_baselines.py --pretrain_path ./Model/checkpoint.pt
  python compare_baselines.py --temp 0.7 --top_p 0.9     ← mode créatif
"""

import os
import sys
import json
import argparse
import textwrap
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────
# REGISTRE DES MODÈLES
# ─────────────────────────────────────────────────────────────

# Modèles HuggingFace — chargés via AutoModelForCausalLM
HF_MODELS = {
    "cosmo": {
        "hf_id"  : "HuggingFaceTB/cosmo-1b",
        "label"  : "Cosmo-1B",
        "params" : "1B",
        "notes"  : "Même corpus Cosmopedia — référence directe",
        "tok_id" : "HuggingFaceTB/cosmo-1b",
    },
    "smollm2": {
        "hf_id"  : "HuggingFaceTB/SmolLM2-135M",
        "label"  : "SmolLM2-135M",
        "params" : "135M",
        "notes"  : "SmolLM2 pretrain-only — même famille cosmo2",
        "tok_id" : "HuggingFaceTB/SmolLM2-135M",
    },
    "pythia160": {
        "hf_id"  : "EleutherAI/pythia-160m",
        "label"  : "Pythia-160M",
        "params" : "160M",
        "notes"  : "Même taille que Naylis — baseline industrie (pretrain only)",
        "tok_id" : "EleutherAI/pythia-160m",
    },
    "pythia160d": {
        "hf_id"  : "EleutherAI/pythia-160m-deduped",
        "label"  : "Pythia-160M-deduped",
        "params" : "160M",
        "notes"  : "Même taille, corpus dédupliqué (pretrain only)",
        "tok_id" : "EleutherAI/pythia-160m-deduped",
    },
    "pythia410": {
        "hf_id"  : "EleutherAI/pythia-410m",
        "label"  : "Pythia-410M",
        "params" : "410M",
        "notes"  : "2.5× plus grand que Naylis — plafond Pythia pretrain only",
        "tok_id" : "EleutherAI/pythia-410m",
    },
}

# Modèles Naylis — chargés depuis checkpoint .pt local via NaylisGPT
NAYLIS_MODELS = {
    "naylis_pt": {
        "label"    : "Naylis-160M (pretrain)",
        "params"   : "~160M",
        "notes"    : "Ton pretrain Cosmopedia — sans SFT",
        "instruct" : False,
    },
    "naylis_sft": {
        "label"    : "Naylis-160M (SFT)",
        "params"   : "~160M",
        "notes"    : "Ton modèle SFT fine-tuné — smoltalk/mix",
        "instruct" : True,
    },
}

# Registre unifié (pour display et récap)
ALL_MODELS = {**HF_MODELS, **NAYLIS_MODELS}

# Architecture Naylis — doit correspondre à ton pretrain
NAYLIS_CONFIG = {
    "vocab_size"            : 49_152,
    "embed_dim"             : 768,
    "num_heads"             : 12,
    "num_layers"            : 18,
    "max_seq_len"           : 1024,
    "dropout"               : 0.0,
    "use_rope"              : True,
    "use_yarn"              : False,
    "yarn_scale"            : 1.0,
    "yarn_original_max_len" : 1024,
    "use_swiglu"            : True,
    "n_kv_heads"            : 4,
    "use_qk_norm"           : True,
    "soft_cap"              : None,
    "use_flash_attn"        : True,
    "rel_rank"              : 32,
}

NAYLIS_TOKENIZER_ID = "HuggingFaceTB/cosmo2-tokenizer"
NAYLIS_SYSTEM_PROMPT = "You are Naylis, a helpful and knowledgeable AI assistant."

# ─────────────────────────────────────────────────────────────
# PROMPTS DE TEST
# ─────────────────────────────────────────────────────────────

# Mode completion (pretrain-style) — pour tous les modèles
PRETRAIN_PROMPTS = [
    ("capital_france",  "The capital of France is"),
    ("capital_japan",   "The capital of Japan is"),
    ("speed_light",     "The speed of light is approximately"),
    ("largest_planet",  "The largest planet in our solar system is"),
    ("telephone",       "The telephone was invented by"),
    ("einstein",        "Albert Einstein was born in"),
    ("water_formula",   "The chemical formula of water is"),
    ("math_simple",     "2 + 2 ="),
    ("math_medium",     "The square root of 144 is"),
    ("sun_star",        "The Sun is a"),
    ("dna",             "DNA stands for"),
    ("python_lang",     "Python is a programming language created by"),
    ("moon_distance",   "The average distance from Earth to the Moon is approximately"),
    ("continents",      "There are 7 continents on Earth. The largest continent is"),
    ("gravity",         "Isaac Newton discovered the law of gravity when"),
]

# Mode instruct (ChatML) — pour Naylis SFT uniquement (section bonus)
INSTRUCT_PROMPTS = [
    ("hello",           "Hello! How are you?"),
    ("capital_france",  "What is the capital of France?"),
    ("math",            "What is 2 + 2?"),
    ("space_fact",      "Tell me a fun fact about space."),
    ("planet",          "What is the largest planet in our solar system?"),
    ("light_speed",     "What is the speed of light?"),
    ("telephone",       "Who invented the telephone?"),
    ("continents",      "How many continents are there on Earth?"),
    ("gravity",         "Who discovered the law of gravity?"),
    ("dna",             "What does DNA stand for?"),
]

# ─────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Compare baseline LMs + Naylis")
    p.add_argument("--models",        default="all",
                   help="Modèles : cosmo,smollm2,pythia160,pythia160d,pythia410,"
                        "naylis_pt,naylis_sft,all  (séparés par virgule)")
    p.add_argument("--pretrain_path", default="./Model/naylis_pretrain.pt",
                   help="Chemin vers le checkpoint pretrain Naylis")
    p.add_argument("--sft_path",      default="./Model/naylis_sft.pt",
                   help="Chemin vers le checkpoint SFT Naylis")
    p.add_argument("--temp",          type=float, default=0.0)
    p.add_argument("--top_k",         type=int,   default=0)
    p.add_argument("--top_p",         type=float, default=1.0)
    p.add_argument("--rep_penalty",   type=float, default=1.0)
    p.add_argument("--max_tokens",    type=int,   default=80)
    p.add_argument("--quick-test",    action="store_true",
                   help="Seulement les 4 premiers prompts par modèle")
    p.add_argument("--device",        default="auto")
    p.add_argument("--no-sft-instruct", action="store_true",
                   help="Désactive la section bonus ChatML pour naylis_sft")
    p.add_argument("--output_json",   default="compare_results.json",
                   help="Fichier JSON de sortie (défaut: compare_results.json)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# IMPORT NAYLIS (lazy — uniquement si naylis_* est demandé)
# ─────────────────────────────────────────────────────────────
def _import_naylis():
    """Ajoute les paths Core et importe NaylisGPT."""
    _root = os.path.dirname(os.path.abspath(__file__))
    for sub in ("Core/Model", "Core/Attention", "Core/FeedForward", "Core/TransformerBlock"):
        path = os.path.join(_root, sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from HessGpt import NaylisGPT
        return NaylisGPT
    except ImportError as e:
        print(f"\n  ❌ Impossible d'importer NaylisGPT : {e}")
        print("  → Vérifie que Core/Model/HessGpt.py est accessible depuis ce répertoire.")
        raise


# ─────────────────────────────────────────────────────────────
# GÉNÉRATION — HuggingFace (avec KV cache)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_hf(
    model, tokenizer, prompt, device,
    temp=0.0, top_k=0, top_p=1.0, rep_penalty=1.0,
    max_tokens=80, max_ctx=1024,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(ids) > max_ctx - max_tokens:
        ids = ids[-(max_ctx - max_tokens):]

    eos_id    = tokenizer.eos_token_id or 0
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []

    out      = model(input_ids, use_cache=True)
    past_kv  = out.past_key_values
    next_logits = out.logits[0, -1, :].float()

    for _ in range(max_tokens):
        logits = next_logits.clone()

        if rep_penalty != 1.0:
            for tok_id in set(ids) | set(generated):
                if 0 <= tok_id < logits.shape[0]:
                    logits[tok_id] /= rep_penalty if logits[tok_id] > 0 else 1 / rep_penalty

        if temp == 0.0:
            next_token = logits.argmax().item()
        else:
            logits = logits / temp
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < topk_vals[-1], float('-inf'))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove    = (cum_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
                sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
                logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)
            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        if next_token == eos_id:
            break
        generated.append(next_token)

        tok_t   = torch.tensor([[next_token]], dtype=torch.long, device=device)
        out     = model(tok_t, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_logits = out.logits[0, -1, :].float()

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────
# GÉNÉRATION — Naylis (forward complet, sans KV cache HF)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_naylis(
    model, tokenizer, prompt, device,
    temp=0.0, top_k=0, top_p=1.0, rep_penalty=1.0,
    max_tokens=80, max_ctx=1024,
) -> str:
    """
    Génération pour NaylisGPT — forward sur la séquence complète à chaque step.
    Plus lent que KV cache mais robuste à tous les modèles custom.
    """
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(ids) > max_ctx - max_tokens:
        ids = ids[-(max_ctx - max_tokens):]

    eos_id    = tokenizer.eos_token_id or 0
    generated = []

    for _ in range(max_tokens):
        seq       = ids + generated
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)

        # NaylisGPT peut retourner : (logits,) / logits / objet avec .logits
        out = model(input_ids)
        if isinstance(out, torch.Tensor):
            logits = out[0, -1, :].float()
        elif isinstance(out, tuple):
            logits = out[0][0, -1, :].float()
        else:
            logits = out.logits[0, -1, :].float()

        if rep_penalty != 1.0:
            for tok_id in set(seq):
                if 0 <= tok_id < logits.shape[0]:
                    if logits[tok_id] > 0:
                        logits[tok_id] /= rep_penalty
                    else:
                        logits[tok_id] *= rep_penalty

        if temp == 0.0:
            next_token = logits.argmax().item()
        else:
            logits = logits / temp
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < topk_vals[-1], float('-inf'))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove    = (cum_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
                sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
                logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)
            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        if next_token == eos_id:
            break
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _format_chatml(prompt: str) -> str:
    """Formatte un prompt en ChatML pour Naylis SFT."""
    return (
        f"<|im_start|>system\n{NAYLIS_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─────────────────────────────────────────────────────────────
# CHARGEMENT — HuggingFace
# ─────────────────────────────────────────────────────────────
def load_hf_model(model_key: str, device: str):
    info  = HF_MODELS[model_key]
    print(f"\n  Chargement : {info['hf_id']}")
    print(f"  ({info['notes']})")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(info["tok_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        info["hf_id"],
        dtype             = dtype,
        trust_remote_code = True,
    )
    model = model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✅ {params:.0f}M params — dtype={dtype}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# CHARGEMENT — Naylis (.pt checkpoint)
# ─────────────────────────────────────────────────────────────
def load_naylis_model(model_key: str, checkpoint_path: str, device: str):
    info = NAYLIS_MODELS[model_key]
    print(f"\n  Chargement Naylis : {checkpoint_path}")
    print(f"  ({info['notes']})")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}\n"
            f"  → Lance d'abord pretrain.py / sft.py"
        )

    NaylisGPT = _import_naylis()

    # Charge le checkpoint (CPU d'abord pour éviter les surprises VRAM)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Récupère la config depuis le checkpoint si disponible,
    # sinon utilise NAYLIS_CONFIG par défaut
    cfg = ckpt.get("config", NAYLIS_CONFIG)

    # Extrait uniquement les clés connues de NaylisGPT
    model_cfg = {k: cfg.get(k, NAYLIS_CONFIG[k]) for k in NAYLIS_CONFIG}

    model = NaylisGPT(**model_cfg)

    # Charge les poids — cherche dans l'ordre les clés connues du checkpoint
    # pretrain.py sauvegarde sous "model_state_dict", sft.py idem
    state_dict = (
        ckpt.get("model_state_dict")   # ← pretrain.py / sft.py  (priorité)
        or ckpt.get("model")
        or ckpt.get("state_dict")
        or ckpt                        # checkpoint = state_dict direct
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠  Clés manquantes ({len(missing)}) : {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠  Clés inattendues ({len(unexpected)}) : {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(dtype).to(device)
    model.eval()

    # Tokenizer Naylis = cosmo2
    tokenizer = AutoTokenizer.from_pretrained(NAYLIS_TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✅ {params:.0f}M params — dtype={dtype}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# RUN — un modèle sur une liste de prompts
# ─────────────────────────────────────────────────────────────
def run_model(model_key, model, tokenizer, prompts, args, device,
              is_naylis=False, chatml=False):
    info   = ALL_MODELS[model_key]
    wrap_w = 70

    mode_label = " [ChatML]" if chatml else " [completion]"
    print("\n" + "█" * 65)
    print(f"  {info['label']}  [{info['params']}]{mode_label}")
    print(f"  {info['notes']}")
    print("█" * 65)

    results = []
    gen_fn  = generate_naylis if is_naylis else generate_hf

    for name, prompt_text in prompts:
        print(f"\n  [{name}]")

        if chatml:
            prompt_input = _format_chatml(prompt_text)
            print(f"  PROMPT (instruct) : {prompt_text}")
        else:
            prompt_input = prompt_text
            print(f"  PROMPT : {prompt_text}")

        response = gen_fn(
            model, tokenizer, prompt_input, device,
            temp       = args.temp,
            top_k      = args.top_k,
            top_p      = args.top_p,
            rep_penalty= args.rep_penalty,
            max_tokens = args.max_tokens,
        )

        wrapped = textwrap.fill(response, width=wrap_w,
                                initial_indent="  OUTPUT : ",
                                subsequent_indent="           ")
        print(wrapped)
        results.append((name, prompt_text, response))

    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = get_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Sélection des modèles
    valid_keys = list(ALL_MODELS.keys())
    if args.models.strip().lower() == "all":
        model_keys = valid_keys
    else:
        model_keys = [m.strip().lower() for m in args.models.split(",")]
        invalid = [m for m in model_keys if m not in ALL_MODELS]
        if invalid:
            print(f"  ❌ Modèles inconnus : {invalid}")
            print(f"  Disponibles : {valid_keys}")
            sys.exit(1)

    # Chemins Naylis
    naylis_paths = {
        "naylis_pt" : args.pretrain_path,
        "naylis_sft": args.sft_path,
    }

    prompts = PRETRAIN_PROMPTS
    if args.quick_test:
        prompts = prompts[:4]

    print("\n" + "=" * 65)
    print("  Baseline Comparison — Naylis vs HuggingFace models")
    print("=" * 65)
    print(f"  Device     : {device}")
    if device == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM       : {vram:.0f} GB")
    print(f"  Modèles    : {model_keys}")
    print(f"  Prompts    : {len(prompts)}")
    mode_str = "greedy (temp=0)" if args.temp == 0.0 else f"sampling (temp={args.temp})"
    print(f"  Mode       : {mode_str}  top_k={args.top_k}  top_p={args.top_p}  rep_penalty={args.rep_penalty}")
    print(f"  Max tokens : {args.max_tokens}")
    print("=" * 65)

    all_results = {}

    for key in model_keys:
        try:
            is_naylis = key in NAYLIS_MODELS

            if is_naylis:
                model, tokenizer = load_naylis_model(key, naylis_paths[key], device)
            else:
                model, tokenizer = load_hf_model(key, device)

            results = run_model(key, model, tokenizer, prompts, args, device,
                                is_naylis=is_naylis, chatml=False)
            all_results[key] = results

            # Section bonus : Naylis SFT en mode instruct (ChatML)
            if key == "naylis_sft" and not args.no_sft_instruct:
                instruct_prompts = INSTRUCT_PROMPTS
                if args.quick_test:
                    instruct_prompts = instruct_prompts[:4]
                print("\n" + "─" * 65)
                print("  [BONUS] Naylis SFT — mode instruct (ChatML)")
                print("─" * 65)
                run_model(key, model, tokenizer, instruct_prompts, args, device,
                          is_naylis=True, chatml=True)

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"\n  ❌ {key} échoué : {e}")
            traceback.print_exc()

    # ── Récap comparatif côte à côte ─────────────────────────
    if len(all_results) > 1:
        print("\n\n" + "=" * 65)
        print("  RÉCAP COMPARATIF  (mode completion — même prompt pour tous)")
        print("=" * 65)

        all_names = []
        seen_names = set()
        for results in all_results.values():
            for name, _, _ in results:
                if name not in seen_names:
                    all_names.append(name)
                    seen_names.add(name)

        col_w = 10
        for name in all_names:
            print(f"\n  [{name}]")
            for key in model_keys:
                if key not in all_results:
                    continue
                label  = ALL_MODELS[key]["params"]
                tag    = ALL_MODELS[key]["label"]
                match  = next((r for r in all_results[key] if r[0] == name), None)
                if match:
                    resp = match[2][:110] + "..." if len(match[2]) > 110 else match[2]
                    print(f"  {tag:<28} : {resp}")

    # ── Export JSON ───────────────────────────────────────────
    if all_results:
        json_out = {
            "metadata": {
                "date"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device"     : device,
                "mode"       : mode_str,
                "temp"       : args.temp,
                "top_k"      : args.top_k,
                "top_p"      : args.top_p,
                "rep_penalty": args.rep_penalty,
                "max_tokens" : args.max_tokens,
                "models_run" : model_keys,
            },
            "results": {},
        }
        for key, results in all_results.items():
            info = ALL_MODELS[key]
            json_out["results"][key] = {
                "label" : info["label"],
                "params": info["params"],
                "notes" : info["notes"],
                "prompts": [
                    {"name": name, "prompt": prompt, "output": output}
                    for name, prompt, output in results
                ],
            }

        out_path = args.output_json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 Résultats sauvegardés → {out_path}")

    print("\n" + "=" * 65)
    print("  Terminé.")
    print("=" * 65)
    print()
    print("  Interprétation :")
    print("    - Cosmo-1B   >> Naylis       → normal (6× plus grand)")
    print("    - SmolLM2-135M < Naylis      → bon signe (taille similaire, ton corpus)")
    print("    - Pythia-160M < Naylis_pt    → ton pretrain surpasse le baseline industrie")
    print("    - Naylis_sft > Naylis_pt     → le SFT améliore les réponses")
    print("=" * 65)


if __name__ == "__main__":
    main()
