#!/usr/bin/env python3
"""Total VLM deep extraction: Qwen3-VL-8B + Llama-3.2-Vision-11B.

6-pass extraction per image — EVERYTHING extractable is saved.

Pass 1: Generation (3 prompts × text + token probs + top-k + score logits)
Pass 2: Attention + Hidden States (per-head entropy, spatial heatmap, rollout,
        all-layer hidden states, cosine sim, intrinsic dim)
Pass 3: Encoding Cost (multiple refs, per-token loss, image/text separation)
Pass 4: Gradient Attribution (per-digit gradients, patch heatmap)
Pass 5: Vision Encoder (patch embeddings, layer-wise features)
Pass 6: Cross-Attention (Llama only)

All raw data saved as JSON + .npy for offline analysis.
"""

import torch
import numpy as np
import json
import re
import argparse
import time
import gc
from pathlib import Path
from PIL import Image
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "qwen3vl": {
        "id": "Qwen/Qwen3-VL-8B-Instruct",
        "type": "qwen3",
    },
    "llama32v": {
        "id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "type": "llama",
    },
}

PROMPTS = OrderedDict([
    ("aesthetic", (
        "Rate the aesthetic quality of this artwork on a scale of 1 to 10, "
        "where 1 is very low quality and 10 is masterpiece-level. "
        "Consider composition, technique, emotional impact, cultural significance, "
        "use of space, color, and artistic mastery. "
        "First line: just the number. Then explain your reasoning in detail."
    )),
    ("structure", (
        "Analyze the visual and compositional structure of this artwork in detail. "
        "Discuss: spatial organization, use of negative space vs filled space, "
        "color relationships, line quality, texture, depth cues, "
        "rhythmic patterns, and any mathematical proportions you observe."
    )),
    ("cultural", (
        "Identify what cultural or artistic tradition this artwork belongs to. "
        "Describe the specific aesthetic principles and techniques characteristic "
        "of that tradition visible in this work. Be precise and detailed."
    )),
])

ENCODING_REFS = [
    "This is a painting.",
    "This is a beautiful masterpiece.",
    "This is an ugly, poorly made artwork.",
    "Describe every detail of this image.",
]

TEMPERATURES = [0.0, 0.5, 1.0]


def load_model(model_name, gpu_id=0):
    cfg = MODELS[model_name]
    model_id = cfg["id"]
    n_gpus = torch.cuda.device_count()
    print(f"[GPU {gpu_id}] Loading {model_id} (visible GPUs: {n_gpus})...", flush=True)

    if n_gpus >= 2:
        max_mem = {i: "45GiB" for i in range(n_gpus)}
        dm = "auto"
        print(f"  Multi-GPU mode: splitting across {n_gpus} GPUs", flush=True)
    else:
        max_mem = None
        dm = f"cuda:{gpu_id}"

    if cfg["type"] == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        kwargs = dict(torch_dtype=torch.bfloat16, device_map=dm, attn_implementation="eager")
        if max_mem:
            kwargs["max_memory"] = max_mem
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        processor = AutoProcessor.from_pretrained(model_id)
    elif cfg["type"] == "llama":
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        kwargs = dict(torch_dtype=torch.bfloat16, device_map=dm, attn_implementation="eager")
        if max_mem:
            kwargs["max_memory"] = max_mem
        model = MllamaForConditionalGeneration.from_pretrained(model_id, **kwargs)
        processor = AutoProcessor.from_pretrained(model_id)

    model.eval()
    print(f"[GPU {gpu_id}] Model loaded. Device map: {getattr(model, 'hf_device_map', 'single')}", flush=True)
    return model, processor, cfg["type"]


def _prepare_inputs(processor, model_type, img, prompt, device, assistant_text=None):
    if model_type == "qwen3":
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        if assistant_text:
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": assistant_text},
            ]})
        inputs = processor.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=(assistant_text is None),
            return_dict=True, return_tensors="pt"
        ).to(device)
    elif model_type == "llama":
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        if assistant_text:
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": assistant_text},
            ]})
        text_input = processor.apply_chat_template(
            messages, add_generation_prompt=(assistant_text is None))
        inputs = processor(img, text_input, return_tensors="pt").to(device)
    return inputs


def _get_digit_token_ids(processor):
    """Get token IDs for digits 1-10 for score extraction."""
    digit_ids = {}
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    for d in range(1, 11):
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        if toks:
            digit_ids[d] = toks[0]
    return digit_ids


# ─────────────────── PASS 1: GENERATION ───────────────────

def pass1_generate(model, processor, model_type, img, prompt, top_k_save=50):
    """Generate text + full token-level probability data."""
    inputs = _prepare_inputs(processor, model_type, img, prompt, model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            output_scores=True, output_attentions=False,
            output_hidden_states=False, return_dict_in_generate=True,
        )

    gen_ids = gen_out.sequences[:, input_len:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    scores_tensor = torch.stack(gen_out.scores, dim=1)
    n_gen = min(gen_ids.shape[1], scores_tensor.shape[1])

    log_probs = torch.log_softmax(scores_tensor, dim=-1)
    token_lps = log_probs[0, :n_gen].gather(
        1, gen_ids[0, :n_gen].unsqueeze(-1)).squeeze(-1)

    probs = torch.softmax(scores_tensor[0, :n_gen], dim=-1)

    topk_vals, topk_ids = probs.topk(top_k_save, dim=-1)
    topk_data = {
        "ids": topk_ids.cpu().tolist(),
        "probs": topk_vals.cpu().half().tolist(),
    }

    prob_ent = float((-probs * (probs + 1e-10).log()).sum(-1).mean().item())
    top1 = float(topk_vals[:, 0].mean().item())
    top5 = float(topk_vals[:, :5].sum(-1).mean().item())
    kurt = float(_kurtosis(topk_vals[:, 0]).item()) if n_gen > 3 else 0.0

    digit_ids = _get_digit_token_ids(processor)
    score_logits = {}
    if digit_ids and n_gen > 0:
        first_logits = scores_tensor[0, 0]
        for d, tid in digit_ids.items():
            score_logits[str(d)] = float(first_logits[tid].item())

    result = {
        "text": text,
        "gen_perplexity": float(torch.exp(-token_lps.mean()).item()),
        "avg_logprob": float(token_lps.mean().item()),
        "prob_entropy": prob_ent,
        "prob_top1_confidence": top1,
        "prob_top5_mass": top5,
        "prob_kurtosis": kurt,
        "n_tokens_generated": int(n_gen),
        "token_logprobs": token_lps.cpu().tolist(),
        "score_logit_distribution": score_logits,
        "_topk_data": topk_data,
    }

    del gen_out, scores_tensor, log_probs, probs, topk_vals, topk_ids
    torch.cuda.empty_cache()
    return result


def pass1_temperature(model, processor, model_type, img, prompt, temperatures):
    """Score sensitivity across temperatures."""
    results = {}
    for temp in temperatures:
        inputs = _prepare_inputs(processor, model_type, img, prompt, model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            if temp == 0.0:
                gen_out = model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                    return_dict_in_generate=True)
            else:
                gen_out = model.generate(
                    **inputs, max_new_tokens=64, do_sample=True,
                    temperature=temp, return_dict_in_generate=True)
        gen_ids = gen_out.sequences[:, input_len:]
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        score = extract_score(text)
        results[f"temp_{temp}"] = {"score": score, "text_preview": text[:100]}
        del gen_out
        torch.cuda.empty_cache()
    return results


# ─────────────────── PASS 2: ATTENTION + HIDDEN STATES ───────────────────

def pass2_attention_hidden(model, processor, model_type, img, prompt, response_text):
    """Full attention + hidden state extraction."""
    inputs = _prepare_inputs(
        processor, model_type, img, prompt, model.device,
        assistant_text=response_text)
    seq_len = inputs["input_ids"].shape[1]

    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"pass2_oom": True}

    result = {}

    # ── Attention analysis ──
    if hasattr(outputs, 'attentions') and outputs.attentions:
        attn_tuple = outputs.attentions
        n_layers = len(attn_tuple)
        per_head_entropy = []
        per_head_max = []
        image_text_attn_ratio = []

        for layer_attn in attn_tuple:
            aw = layer_attn[0].float()
            n_heads = aw.shape[0]
            head_ents = []
            head_maxes = []
            for h in range(n_heads):
                row = aw[h, -1, :]
                p = row / (row.sum() + 1e-10)
                ent = float(-(p * (p + 1e-10).log()).sum().item())
                head_ents.append(ent)
                head_maxes.append(float(p.max().item()))
            per_head_entropy.append(head_ents)
            per_head_max.append(head_maxes)

            avg_row = aw[:, -1, :].mean(dim=0)
            avg_p = avg_row / (avg_row.sum() + 1e-10)
            mid = seq_len // 2
            img_attn = float(avg_p[:mid].sum().item())
            txt_attn = float(avg_p[mid:].sum().item())
            image_text_attn_ratio.append(
                img_attn / (txt_attn + 1e-10))

        result["attn_per_head_entropy"] = per_head_entropy
        result["attn_per_head_max"] = per_head_max
        result["attn_image_text_ratio_per_layer"] = image_text_attn_ratio

        layer_mean_ent = [float(np.mean(le)) for le in per_head_entropy]
        result["attn_entropy_per_layer"] = layer_mean_ent
        result["attn_entropy_mean"] = float(np.mean(layer_mean_ent))
        result["attn_entropy_std"] = float(np.std(layer_mean_ent))
        n3 = n_layers // 3
        result["attn_entropy_early"] = float(np.mean(layer_mean_ent[:n3])) if n3 else 0
        result["attn_entropy_mid"] = float(np.mean(layer_mean_ent[n3:2*n3])) if n3 else 0
        result["attn_entropy_late"] = float(np.mean(layer_mean_ent[2*n3:])) if n3 else 0

        try:
            last_attn = attn_tuple[-1][0].float()
            spatial_map = last_attn[:, -1, :].mean(dim=0).detach().cpu().numpy()
            result["_attn_spatial_map"] = spatial_map.astype(np.float16)
        except:
            pass

        try:
            rollout = torch.eye(seq_len, device=attn_tuple[0].device, dtype=torch.float32)
            for layer_attn in attn_tuple:
                aw = layer_attn[0].float().mean(dim=0)
                if aw.shape[-1] != seq_len or aw.shape[-2] != seq_len:
                    continue
                aw = aw / (aw.sum(dim=-1, keepdim=True) + 1e-10)
                rollout = aw @ rollout
            rollout_last = rollout[-1].detach().cpu().numpy()
            result["_attn_rollout"] = rollout_last.astype(np.float16)
        except Exception as e:
            result["attn_rollout_error"] = str(e)[:100]

    # ── Hidden states ──
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        hs_tuple = outputs.hidden_states
        n_hs = len(hs_tuple)

        all_layer_pooled = []
        layer_norms = []
        for hs in hs_tuple:
            vec = hs[0].float().mean(dim=0).detach().cpu().numpy()
            all_layer_pooled.append(vec.astype(np.float16))
            layer_norms.append(float(np.linalg.norm(vec)))

        result["hidden_layer_norms"] = layer_norms
        result["_hidden_all_layers_pooled"] = np.stack(all_layer_pooled)

        early_idx = n_hs // 4
        mid_idx = n_hs // 2
        late_idx = n_hs - 1
        key_layers = {}
        for name, idx in [("early", early_idx), ("mid", mid_idx), ("late", late_idx)]:
            full_hs = hs_tuple[idx][0].float().detach().cpu().numpy().astype(np.float16)
            key_layers[name] = full_hs
            result[f"hidden_{name}_layer_idx"] = idx
        result["_hidden_key_layers"] = key_layers

        cosine_sims = []
        for i in range(1, n_hs):
            v1 = hs_tuple[i-1][0].float().detach().mean(dim=0)
            v2 = hs_tuple[i][0].float().detach().mean(dim=0)
            cos = float(torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)).item())
            cosine_sims.append(cos)
        result["hidden_layer_cosine_sim"] = cosine_sims

        try:
            last_hs = hs_tuple[-1][0].float().detach()
            centered = last_hs - last_hs.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(centered)
            s_norm = s / (s.sum() + 1e-10)
            intdim = float(torch.exp(-(s_norm * (s_norm + 1e-10).log()).sum()).item())
            result["hidden_intrinsic_dim"] = intdim
        except:
            pass

        last_vec = all_layer_pooled[-1]
        result["hidden_state_dim"] = int(last_vec.shape[0])
        result["hidden_state_norm"] = float(np.linalg.norm(last_vec))

    del outputs
    torch.cuda.empty_cache()
    return result


# ─────────────────── PASS 3: ENCODING COST ───────────────────

def pass3_encoding_cost(model, processor, model_type, img, ref_texts=ENCODING_REFS):
    """Encoding cost with multiple reference texts + per-token detail."""
    results = {}

    for ref_idx, ref_text in enumerate(ref_texts):
        inputs = _prepare_inputs(
            processor, model_type, img,
            "Describe this artwork.", model.device,
            assistant_text=ref_text)

        labels = inputs["input_ids"].clone()
        vocab_size = getattr(model.config, 'vocab_size', 128256)
        labels[(labels < 0) | (labels >= vocab_size)] = -100
        if hasattr(processor, 'tokenizer'):
            pad_id = getattr(processor.tokenizer, 'pad_token_id', None)
            if pad_id is not None:
                labels[labels == pad_id] = -100

        with torch.no_grad():
            outputs = model(**inputs, labels=labels, output_hidden_states=True)

        total_loss = float(outputs.loss.item())

        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.shape)

        valid_mask = shift_labels != -100
        token_losses = per_token_loss[0][valid_mask[0]].cpu().tolist()

        layer_norms = []
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            for hs in outputs.hidden_states:
                layer_norms.append(float(hs[0].float().norm().item()))

        seq_total = shift_labels.shape[1]
        mid_point = seq_total // 2
        first_half_mask = valid_mask[0].clone()
        first_half_mask[mid_point:] = False
        second_half_mask = valid_mask[0].clone()
        second_half_mask[:mid_point] = False
        img_region_loss = float(per_token_loss[0][first_half_mask].mean().item()) if first_half_mask.any() else 0.0
        txt_region_loss = float(per_token_loss[0][second_half_mask].mean().item()) if second_half_mask.any() else 0.0

        ref_key = f"ref{ref_idx}"
        results[f"encoding_cost_{ref_key}"] = total_loss
        results[f"encoding_per_token_{ref_key}"] = token_losses
        results[f"encoding_layer_norms_{ref_key}"] = layer_norms
        results[f"encoding_img_region_loss_{ref_key}"] = img_region_loss
        results[f"encoding_txt_region_loss_{ref_key}"] = txt_region_loss

        del outputs, logits, per_token_loss
        torch.cuda.empty_cache()

    results["encoding_cost"] = results.get("encoding_cost_ref0", 0)
    results["encoding_img_loss"] = results.get("encoding_img_region_loss_ref0", 0)
    results["encoding_txt_loss"] = results.get("encoding_txt_region_loss_ref0", 0)
    return results


# ─────────────────── PASS 4: GRADIENT ATTRIBUTION ───────────────────

def pass4_gradient(model, processor, model_type, img, prompt):
    """Gradient attribution w.r.t. input embeddings for each score digit.
    Llama 11B: uses 256px to avoid backward OOM on 80GB GPU.
    Qwen 8B: uses full 512px.
    """
    grad_img = img
    if model_type == "llama":
        grad_img = img.copy()
        grad_img.thumbnail((256, 256), Image.LANCZOS)
    inputs = _prepare_inputs(processor, model_type, grad_img, prompt, model.device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    digit_ids = _get_digit_token_ids(processor)

    result = {}
    try:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

        embeds = model.get_input_embeddings()(input_ids)
        embeds = embeds.detach().requires_grad_(True)
        new_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
        new_inputs["inputs_embeds"] = embeds

        outputs = model(**new_inputs)
        last_logits = outputs.logits[0, -1, :]

        all_digit_tids = list(digit_ids.values())
        if all_digit_tids:
            target = last_logits[all_digit_tids].sum()
        else:
            target = last_logits.max()
        target.backward(retain_graph=True)

        grad = embeds.grad[0].float()
        grad_norm = grad.norm(dim=-1)
        total_gn = grad_norm.sum() + 1e-10

        result["grad_attribution_image_frac"] = float(
            grad_norm[:input_len].sum() / total_gn)
        result["grad_norm_mean"] = float(grad_norm.mean().item())
        result["grad_norm_std"] = float(grad_norm.std().item())
        result["grad_norm_max"] = float(grad_norm.max().item())
        result["grad_resolution"] = "256px" if model_type == "llama" else "512px"
        result["_grad_per_token"] = grad_norm.cpu().numpy().astype(np.float16)

        model.zero_grad()
        if embeds.grad is not None:
            embeds.grad.zero_()

        per_digit_grads = {}
        for d, tid in digit_ids.items():
            try:
                if embeds.grad is not None:
                    embeds.grad.zero_()
                score_logit = last_logits[tid]
                score_logit.backward(retain_graph=True)
                g = embeds.grad[0].float().norm(dim=-1)
                per_digit_grads[str(d)] = float(g.mean().item())
            except:
                pass
        result["grad_per_digit_mean"] = per_digit_grads

        del outputs, embeds, grad
        model.gradient_checkpointing_disable()
        torch.cuda.empty_cache()

    except Exception as e:
        result["grad_error"] = str(e)
        try:
            model.gradient_checkpointing_disable()
            torch.cuda.empty_cache()
        except:
            pass
    return result


# ─────────────────── PASS 5: VISION ENCODER ───────────────────

def pass5_vision_encoder(model, processor, model_type, img):
    """Extract vision encoder embeddings."""
    result = {}
    try:
        if model_type == "qwen3":
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe."},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)

            vision_model = None
            if hasattr(model, 'model') and hasattr(model.model, 'visual'):
                vision_model = model.model.visual
            elif hasattr(model, 'visual'):
                vision_model = model.visual

            if vision_model is not None and 'pixel_values' in inputs:
                    with torch.no_grad():
                        vis_out = vision_model(
                            inputs['pixel_values'],
                            output_hidden_states=True)
                    if hasattr(vis_out, 'last_hidden_state'):
                        patch_emb = vis_out.last_hidden_state[0].float().detach().cpu().numpy()
                        result["_vision_patch_embeddings"] = patch_emb.astype(np.float16)
                        result["vision_patch_count"] = int(patch_emb.shape[0])
                        result["vision_embed_dim"] = int(patch_emb.shape[1])
                        result["vision_embed_norm"] = float(np.linalg.norm(patch_emb, axis=1).mean())

                    if hasattr(vis_out, 'hidden_states') and vis_out.hidden_states:
                        vis_layer_norms = []
                        for vhs in vis_out.hidden_states:
                            vis_layer_norms.append(float(vhs[0].float().norm().item()))
                        result["vision_layer_norms"] = vis_layer_norms

        elif model_type == "llama":
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": "Describe."},
            ]}]
            text_input = processor.apply_chat_template(
                messages, add_generation_prompt=True)
            inputs = processor(img, text_input, return_tensors="pt").to(model.device)

            vision_model = None
            if hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
                vision_model = model.model.vision_model
            elif hasattr(model, 'vision_tower'):
                vision_model = model.vision_tower

            if vision_model is not None and 'pixel_values' in inputs:
                with torch.no_grad():
                    pv = inputs['pixel_values']
                    aspect = inputs.get('aspect_ratio_ids', None)
                    aspect_mask = inputs.get('aspect_ratio_mask', None)
                    vis_kwargs = {'output_hidden_states': True}
                    if aspect is not None:
                        vis_kwargs['aspect_ratio_ids'] = aspect
                    if aspect_mask is not None:
                        vis_kwargs['aspect_ratio_mask'] = aspect_mask
                    try:
                        vis_out = vision_model(pv, **vis_kwargs)
                    except TypeError:
                        vis_out = vision_model(pv, output_hidden_states=True)

                if hasattr(vis_out, 'last_hidden_state') and vis_out.last_hidden_state is not None:
                    patch_emb = vis_out.last_hidden_state
                    if patch_emb.dim() == 4:
                        patch_emb = patch_emb.reshape(
                            patch_emb.shape[0], -1, patch_emb.shape[-1])
                    patch_emb = patch_emb[0].float().detach().cpu().numpy()
                    result["_vision_patch_embeddings"] = patch_emb.astype(np.float16)
                    result["vision_patch_count"] = int(patch_emb.shape[0])
                    result["vision_embed_dim"] = int(patch_emb.shape[1])
                    result["vision_embed_norm"] = float(
                        np.linalg.norm(patch_emb, axis=1).mean())
                elif isinstance(vis_out, tuple) and len(vis_out) > 0:
                    patch_emb = vis_out[0]
                    if patch_emb.dim() >= 2:
                        patch_emb = patch_emb.reshape(-1, patch_emb.shape[-1])
                        patch_emb = patch_emb.float().detach().cpu().numpy()
                        result["_vision_patch_embeddings"] = patch_emb.astype(np.float16)
                        result["vision_patch_count"] = int(patch_emb.shape[0])
                        result["vision_embed_dim"] = int(patch_emb.shape[1])

                if hasattr(vis_out, 'hidden_states') and vis_out.hidden_states:
                    vis_layer_norms = []
                    for vhs in vis_out.hidden_states:
                        t = vhs[0] if vhs.dim() >= 3 else vhs
                        vis_layer_norms.append(float(t.float().detach().norm().item()))
                    result["vision_layer_norms"] = vis_layer_norms

    except Exception as e:
        result["vision_encoder_error"] = str(e)
    finally:
        torch.cuda.empty_cache()

    return result


# ─────────────────── PASS 6: CROSS-ATTENTION (Llama) ───────────────────

def pass6_cross_attention(model, processor, model_type, img, prompt, response_text):
    """Extract cross-attention weights (Llama architecture).
    In MllamaForConditionalGeneration, cross-attention layers are interleaved
    with self-attention in the decoder. The attentions output includes both.
    Cross-attention layers have different key/value dimensions (image features).
    """
    if model_type != "llama":
        return {}

    result = {}
    try:
        inputs = _prepare_inputs(
            processor, model_type, img, prompt, model.device,
            assistant_text=response_text)
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=False,
            )

        if hasattr(outputs, 'attentions') and outputs.attentions:
            self_attn_ents = []
            cross_attn_ents = []
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                aw = layer_attn[0].float()
                n_heads = aw.shape[0]
                is_cross = (aw.shape[-1] != aw.shape[-2])
                head_ents = []
                for h in range(n_heads):
                    row = aw[h, -1, :]
                    p = row / (row.sum() + 1e-10)
                    ent = float(-(p * (p + 1e-10).log()).sum().item())
                    head_ents.append(ent)
                if is_cross:
                    cross_attn_ents.append(head_ents)
                else:
                    self_attn_ents.append(head_ents)

            if cross_attn_ents:
                result["cross_attn_per_head_entropy"] = cross_attn_ents
                result["cross_attn_mean_entropy"] = float(
                    np.mean([np.mean(le) for le in cross_attn_ents]))
                result["cross_attn_layer_count"] = len(cross_attn_ents)

            if not cross_attn_ents:
                result["cross_attn_note"] = "no_asymmetric_attention_found"

        if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
            ca_list = outputs.cross_attentions
            ca_ents = []
            for layer_ca in ca_list:
                ca = layer_ca[0].float()
                head_ents = []
                for h in range(ca.shape[0]):
                    p = ca[h, -1, :]
                    p = p / (p.sum() + 1e-10)
                    ent = float(-(p * (p + 1e-10).log()).sum().item())
                    head_ents.append(ent)
                ca_ents.append(head_ents)
            if ca_ents:
                result["cross_attn_explicit_per_head"] = ca_ents
                result["cross_attn_explicit_mean"] = float(
                    np.mean([np.mean(le) for le in ca_ents]))

    except Exception as e:
        result["cross_attn_error"] = str(e)[:200]
    finally:
        torch.cuda.empty_cache()

    return result


# ─────────────────── UTILITIES ───────────────────

def _kurtosis(x):
    m = x.mean()
    s = x.std()
    if s < 1e-8:
        return torch.tensor(0.0)
    return ((x - m) ** 4).mean() / (s ** 4) - 3


def extract_score(text):
    for line in text.strip().split("\n")[:5]:
        m = re.search(r'\b(\d+(?:\.\d+)?)\s*/?\s*10\b|\b(\d+(?:\.\d+)?)\b', line)
        if m:
            val = float(m.group(1) or m.group(2))
            if 1 <= val <= 10:
                return val
    return None


def compute_response_meta(text, score):
    """Extra response-level metrics."""
    words = text.split()
    unique_words = set(w.lower() for w in words)
    first_line = text.strip().split("\n")[0].strip() if text.strip() else ""
    starts_with_digit = bool(re.match(r'^\d', first_line))
    return {
        "response_word_count": len(words),
        "response_unique_words": len(unique_words),
        "response_vocab_diversity": len(unique_words) / max(len(words), 1),
        "response_char_count": len(text),
        "response_line_count": len(text.strip().split("\n")),
        "response_starts_with_digit": starts_with_digit,
        "response_score_found": score is not None,
    }


def pass_logit_lens(model, processor, model_type, img, prompt):
    """Logit lens: apply LM head at intermediate layers."""
    result = {}
    try:
        inputs = _prepare_inputs(processor, model_type, img, prompt, model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        if not hasattr(outputs, 'hidden_states') or not outputs.hidden_states:
            return result

        lm_head = model.lm_head if hasattr(model, 'lm_head') else None
        if lm_head is None and hasattr(model, 'language_model'):
            lm_head = getattr(model.language_model, 'lm_head', None)
        if lm_head is None:
            return {"logit_lens_error": "no lm_head found"}

        hs_tuple = outputs.hidden_states
        n_layers = len(hs_tuple)
        digit_ids = _get_digit_token_ids(processor)
        layer_predictions = []

        check_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        for layer_idx in check_layers:
            hs = hs_tuple[layer_idx][0, -1:, :]
            with torch.no_grad():
                logits = lm_head(hs.to(lm_head.weight.dtype))
            probs = torch.softmax(logits[0], dim=-1)
            top5_vals, top5_ids = probs.topk(5)

            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            top5_tokens = []
            for tid in top5_ids.tolist():
                try:
                    top5_tokens.append(tokenizer.decode([tid]))
                except:
                    top5_tokens.append(f"[{tid}]")

            digit_probs = {}
            for d, tid in digit_ids.items():
                digit_probs[str(d)] = float(probs[tid].item())

            layer_predictions.append({
                "layer": layer_idx,
                "top5_tokens": top5_tokens,
                "top5_probs": top5_vals.tolist(),
                "digit_probs": digit_probs,
            })

        result["logit_lens"] = layer_predictions
        del outputs
        torch.cuda.empty_cache()
    except Exception as e:
        result["logit_lens_error"] = str(e)
        torch.cuda.empty_cache()
    return result


def save_npy_data(result, entry_name, save_dir):
    """Extract _-prefixed numpy arrays from result, save to disk."""
    saved = {}
    keys_to_remove = []
    for k, v in result.items():
        if k.startswith("_") and isinstance(v, np.ndarray):
            fname = f"{entry_name}_{k[1:]}.npy"
            fpath = save_dir / fname
            np.save(fpath, v)
            saved[k[1:] + "_file"] = str(fpath.name)
            keys_to_remove.append(k)
        elif k.startswith("_") and isinstance(v, dict):
            for subk, subv in v.items():
                if isinstance(subv, np.ndarray):
                    fname = f"{entry_name}_{k[1:]}_{subk}.npy"
                    fpath = save_dir / fname
                    np.save(fpath, subv)
                    saved[f"{k[1:]}_{subk}_file"] = str(fpath.name)
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del result[k]
    result.update(saved)


# ─────────────────── MAIN RUN ───────────────────

def run(model_name, shard_idx=0, total_shards=1, gpu_id=0):
    model, processor, model_type = load_model(model_name, gpu_id)

    meta_path = ROOT / "data" / "experiment_metadata_all.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    if total_shards > 1:
        metadata = [m for i, m in enumerate(metadata)
                     if i % total_shards == shard_idx]

    out_dir = ROOT / "results" / f"{model_name}_shard{shard_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_dir = out_dir / "npy"
    npy_dir.mkdir(exist_ok=True)
    json_path = out_dir / "results.json"
    log_path = out_dir / "run.log"

    existing = {}
    if json_path.exists():
        try:
            for e in json.load(open(json_path)):
                existing[e.get("name", "")] = e
        except:
            pass
    results = list(existing.values())
    done_names = set(existing.keys())

    log_f = open(log_path, "a", encoding="utf-8")
    log_f.write(f"\n=== Run {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"model={model_name} shard={shard_idx}/{total_shards} "
                f"gpu={gpu_id} images={len(metadata)} resume={len(existing)} ===\n")
    log_f.flush()

    for i, entry in enumerate(metadata):
        if entry["name"] in done_names:
            continue

        image_path = str(ROOT / entry["path"])
        if not Path(image_path).exists():
            log_f.write(f"[{i+1}/{len(metadata)}] SKIP {entry['name']} - not found\n")
            log_f.flush()
            continue

        t0 = time.time()
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)

        result = {k: v for k, v in entry.items()}
        aesthetic_text = ""

        # ── Pass 1: Generation (all 3 prompts) ──
        for pname, ptext in PROMPTS.items():
            try:
                p1 = pass1_generate(model, processor, model_type, img, ptext)
                topk = p1.pop("_topk_data", None)
                for k, v in p1.items():
                    result[f"{pname}_{k}"] = v
                if topk is not None:
                    np.savez_compressed(
                        npy_dir / f"{entry['name']}_{pname}_topk.npz",
                        ids=np.array(topk["ids"], dtype=np.int32),
                        probs=np.array(topk["probs"], dtype=np.float16))
                    result[f"{pname}_topk_file"] = f"{entry['name']}_{pname}_topk.npz"
                if pname == "aesthetic":
                    result["aesthetic_score"] = extract_score(p1["text"])
                    aesthetic_text = p1["text"]
                    meta = compute_response_meta(p1["text"], result["aesthetic_score"])
                    for mk, mv in meta.items():
                        result[f"aesthetic_{mk}"] = mv
            except Exception as e:
                result[f"{pname}_error"] = str(e)
                log_f.write(f"ERR[p1_{pname}]: {e}\n")
                torch.cuda.empty_cache()

        # ── Pass 1 ext: Temperature sensitivity ──
        try:
            temp_res = pass1_temperature(
                model, processor, model_type, img,
                PROMPTS["aesthetic"], TEMPERATURES)
            result["temperature_sensitivity"] = temp_res
        except Exception as e:
            result["temperature_error"] = str(e)
            torch.cuda.empty_cache()

        # ── Pass 2: Attention + Hidden States ──
        try:
            p2 = pass2_attention_hidden(
                model, processor, model_type, img,
                PROMPTS["aesthetic"], aesthetic_text)
            save_npy_data(p2, entry["name"], npy_dir)
            for k, v in p2.items():
                result[f"aesthetic_{k}"] = v
        except Exception as e:
            result["pass2_error"] = str(e)
            log_f.write(f"ERR[p2]: {e}\n")
            torch.cuda.empty_cache()

        # ── Logit Lens ──
        try:
            ll = pass_logit_lens(model, processor, model_type, img, PROMPTS["aesthetic"])
            for k, v in ll.items():
                result[f"aesthetic_{k}"] = v
        except Exception as e:
            result["logit_lens_error"] = str(e)
            torch.cuda.empty_cache()

        # ── Pass 3: Encoding Cost ──
        try:
            p3 = pass3_encoding_cost(model, processor, model_type, img)
            for k, v in p3.items():
                result[k] = v
        except Exception as e:
            result["encoding_cost_error"] = str(e)
            log_f.write(f"ERR[p3]: {e}\n")
            try:
                torch.cuda.empty_cache()
            except:
                pass

        # ── Pass 4: Gradient Attribution (LAST - after attn extracted, with checkpointing) ──
        gc.collect()
        torch.cuda.empty_cache()
        try:
            p4 = pass4_gradient(
                model, processor, model_type, img, PROMPTS["aesthetic"])
            grad_arr = p4.pop("_grad_per_token", None)
            for k, v in p4.items():
                result[f"aesthetic_{k}"] = v
            if grad_arr is not None:
                np.save(npy_dir / f"{entry['name']}_grad.npy", grad_arr)
                result["aesthetic_grad_file"] = f"{entry['name']}_grad.npy"
        except Exception as e:
            result["grad_error"] = str(e)
            log_f.write(f"ERR[p4]: {e}\n")
            try:
                torch.cuda.empty_cache()
            except:
                pass

        # ── Pass 5: Vision Encoder ──
        try:
            p5 = pass5_vision_encoder(model, processor, model_type, img)
            vis_emb = p5.pop("_vision_patch_embeddings", None)
            for k, v in p5.items():
                result[k] = v
            if vis_emb is not None:
                np.save(npy_dir / f"{entry['name']}_vision.npy", vis_emb)
                result["vision_embed_file"] = f"{entry['name']}_vision.npy"
        except Exception as e:
            result["vision_error"] = str(e)
            torch.cuda.empty_cache()

        # ── Pass 6: Cross-Attention (Llama only) ──
        if model_type == "llama":
            try:
                p6 = pass6_cross_attention(
                    model, processor, model_type, img,
                    PROMPTS["aesthetic"], aesthetic_text)
                for k, v in p6.items():
                    result[f"aesthetic_{k}"] = v
            except Exception as e:
                result["cross_attn_error"] = str(e)
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        score = result.get("aesthetic_score", "?")
        enc = result.get("encoding_cost", "?")
        has_grad = "grad_attribution_image_frac" in str(result)
        has_attn = "attn_entropy_mean" in str(result)
        result["inference_time_sec"] = round(elapsed, 2)

        results.append(result)

        status = f"score={score} enc={enc} attn={'Y' if has_attn else 'N'} grad={'Y' if has_grad else 'N'}"
        log_f.write(f"[{i+1}/{len(metadata)}] {entry['name']} {status} {elapsed:.1f}s\n")
        log_f.flush()
        print(f"[GPU{gpu_id}][{i+1}/{len(metadata)}] {entry['name']} "
              f"{status} {elapsed:.1f}s", flush=True)

        if (i + 1) % 3 == 0:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

        del img
        gc.collect()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_f.write(f"\n=== Done: {len(results)} results ===\n")
    log_f.close()
    print(f"\n[GPU{gpu_id}] Saved {len(results)} -> {json_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    run(args.model, args.shard, args.total, args.gpu)
