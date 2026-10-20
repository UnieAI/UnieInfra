"""HF reference: per-layer hidden states for one short prompt.

Pairs with `TOKENSPEED_DUMP_HIDDEN` in models/gemma.py; compare with
`gemma_hidden_compare.py`. Loads the text stack only, so the 62 GB of weights
are the only thing on the card -- run it after the engine has released it.

Takes the token ids from the engine's own dump by default. Retokenizing the
same string is not the same experiment: gemma is very sensitive to a missing
BOS, and the two sides disagreeing about one special token looks exactly like a
broken layer.
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/google/gemma-4-31B-it")
    ap.add_argument("--prompt", default="The")
    ap.add_argument("--out", default="/workspace/roy/.hidden-hf.pt")
    # Take the ids from the engine's own dump when it exists. Retokenizing the
    # same string is not the same experiment: gemma is very sensitive to a
    # missing BOS, and the two sides disagreeing about one special token would
    # look exactly like a broken layer.
    ap.add_argument("--ids-from", default="/workspace/roy/.hidden-ours.pt")
    args = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = None
    if args.ids_from:
        try:
            ids = torch.load(args.ids_from, map_location="cpu")["input_ids"]
            ids = [int(x) for x in ids.flatten().tolist()]
            print("input_ids taken from", args.ids_from, ids)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  could not read ids from {args.ids_from}: "
                f"{type(exc).__name__}: {str(exc)[:70]}"
            )
    if ids is None:
        ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
        print("input_ids (retokenized)", ids)

    # gemma-4 is a *ForConditionalGeneration checkpoint, so the CausalLM auto
    # class may refuse it; try the image-text-to-text class before falling back
    # to the plain one. Whichever loads, only the text stack is exercised below.
    model = None
    for cls_name in (
        "AutoModelForImageTextToText",
        "AutoModelForCausalLM",
        "AutoModel",
    ):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                args.model,
                dtype=torch.bfloat16,
                device_map={"": 0},
                attn_implementation="eager",
            )
            print("loaded via", cls_name)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  {cls_name}: {type(exc).__name__}: {str(exc)[:90]}")
    if model is None:
        raise SystemExit("no auto class could load this checkpoint")
    model.eval()
    lm = getattr(model, "model", model)
    text = getattr(lm, "language_model", lm)

    x = torch.tensor([ids], device="cuda")
    rec = {"input_ids": torch.tensor(ids)}
    with torch.no_grad():
        out = text(input_ids=x, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: embedding output, then each layer
    print(f"{len(hs)} hidden states (embed + {len(hs) - 1} layers)")
    rec["embed"] = hs[0][0].float().cpu()
    for i in range(1, len(hs)):
        rec[f"layer{i - 1}"] = hs[i][0].float().cpu()
    rec["final_norm"] = out.last_hidden_state[0].float().cpu()

    head = getattr(model, "lm_head", None) or getattr(lm, "lm_head", None)
    if head is None:
        # Tied embeddings: the head is the embedding matrix transposed.
        emb = text.get_input_embeddings()
        logits = out.last_hidden_state.to(emb.weight.dtype) @ emb.weight.T
    else:
        logits = head(out.last_hidden_state)
    cap = getattr(text.config, "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
    top = logits[0, -1].topk(5)
    rec["top5_ids"] = top.indices.cpu()
    rec["top5_vals"] = top.values.float().cpu()
    print(
        "HF top-5 next tokens:",
        [
            (tok.decode([i]), round(float(v), 3))
            for i, v in zip(top.indices.tolist(), top.values.tolist())
        ],
    )
    torch.save(rec, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
