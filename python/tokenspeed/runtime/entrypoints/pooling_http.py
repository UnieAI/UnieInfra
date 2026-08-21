# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""HTTP for the pooling runtimes: ``/v1/embeddings`` and ``/v1/rerank``.

This server owns its engine rather than proxying, which is the whole reason
it exists. ``control_server.py``'s ``/v1/*`` routes forward to the smg
gateway, and the gateway reaches the engine over
``tokenspeed_scheduler.proto``, whose RPC set is Generate / HealthCheck /
Abort / GetModelInfo / GetServerInfo / GetLoads / FlushCache / profiling /
GetTokenizer / SubscribeKvEvents. There is no Embed. (``tokenspeed_encoder.
proto`` does have an ``Encode``, and it is not this: that servicer is the EPD
vision encode worker -- ``server.py`` calls it "the vision-only encode loop"
-- and it moves image features, not text embeddings. The two share a word.)

So an embedding or reranking model could be *run* -- ``Engine.encode`` is
verified against HuggingFace for six encoder architectures and three
rerankers -- and could not be *reached*. This closes that, in-repo, without
touching the out-of-repo proto or gateway.

Handlers are ``def``, not ``async def``, so FastAPI runs them in its
threadpool and they call ``Engine.encode`` -- the engine's own sync facade,
which owns the loop the tokenizer manager is bound to. Awaiting
``tokenizer_manager.generate_request`` from uvicorn's loop instead looks
tidier and hangs: the request is submitted on one event loop and its
completion is delivered on another. Measured, not reasoned about -- the
first version of this file did exactly that and never answered.

Concurrency still reaches the scheduler: each threadpool worker submits its
own request, and the engine batches whatever is in flight.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import numpy as np
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from tokenspeed.runtime.models.bert import TOKEN_TYPE_SHIFT
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["create_app"]


def _pool(engine, *, text=None, input_ids=None) -> list[list[float]]:
    """One pooled vector per input, through the engine's sync facade."""
    result = engine.encode(prompt=text, input_ids=input_ids)
    if isinstance(result, dict):
        result = [result]
    return [item["embedding"] for item in result]


def _normalise_input(value: Any) -> tuple[list[str] | None, list[list[int]] | None]:
    """Split OpenAI's overloaded ``input`` into text or token ids.

    The field accepts a string, a list of strings, a list of ints, or a list
    of lists of ints, and the last two are only distinguishable by looking
    inside. Getting it wrong turns token ids into the *text* of a number.
    """
    if isinstance(value, str):
        return [value], None
    if not isinstance(value, list) or not value:
        raise ValueError("`input` must be a non-empty string or list")
    if all(isinstance(v, str) for v in value):
        return list(value), None
    if all(isinstance(v, int) for v in value):
        return None, [list(value)]
    if all(isinstance(v, list) and all(isinstance(i, int) for i in v) for v in value):
        return None, [list(v) for v in value]
    raise ValueError(
        "`input` must be a string, a list of strings, a list of ints, or a "
        "list of lists of ints -- not a mixture"
    )


def _encode_embedding(vector: list[float], encoding_format: str):
    if encoding_format == "base64":
        packed = np.asarray(vector, dtype=np.float32).tobytes()
        return base64.b64encode(packed).decode("ascii")
    return vector


def create_app(engine, served_model_name: str) -> FastAPI:
    app = FastAPI(title="TokenSpeed pooling server")
    app.state.engine = engine
    app.state.model = served_model_name

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "tokenspeed",
                }
            ],
        }

    @app.post("/v1/embeddings")
    def embeddings(body: dict = Body(...)):
        try:
            text, input_ids = _normalise_input(body.get("input"))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        encoding_format = body.get("encoding_format", "float")
        if encoding_format not in ("float", "base64"):
            return JSONResponse(
                {"error": f"unsupported encoding_format {encoding_format!r}"},
                status_code=400,
            )

        vectors = _pool(app.state.engine, text=text, input_ids=input_ids)
        return {
            "object": "list",
            "model": body.get("model") or served_model_name,
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": _encode_embedding(v, encoding_format),
                }
                for i, v in enumerate(vectors)
            ],
            # Prompt tokens are not counted here rather than being guessed:
            # the engine tokenized these and this layer did not, and a made-up
            # count is worse than an honest zero for anything billing on it.
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    def _rerank(body: dict):
        query = body.get("query")
        documents = body.get("documents")
        if not isinstance(query, str) or not documents:
            return JSONResponse(
                {"error": "`query` (string) and `documents` (list) are required"},
                status_code=400,
            )
        documents = [d if isinstance(d, str) else d.get("text", "") for d in documents]

        tokenizer = app.state.engine.tokenizer_manager.tokenizer
        if tokenizer is None:
            return JSONResponse(
                {
                    "error": "server started with --skip-tokenizer-init; /v1/rerank "
                    "needs a tokenizer to build the query/document pair"
                },
                status_code=400,
            )

        packed: list[list[int]] = []
        for document in documents:
            encoded = tokenizer(query, document)
            ids = encoded["input_ids"]
            # BERT tells the pair's halves apart with segment ids, and this
            # engine carries them in the token id's high bit rather than in a
            # second per-token tensor (models/bert.py encode_token_type_ids).
            # Models with type_vocab_size <= 1 return zeros here and separate
            # the halves with their separator token instead.
            types = encoded.get("token_type_ids") or [0] * len(ids)
            packed.append([i | (t << TOKEN_TYPE_SHIFT) for i, t in zip(ids, types)])

        scores = _pool(app.state.engine, input_ids=packed)
        # Raw logits, matching AutoModelForSequenceClassification and every
        # one of these checkpoints' own README. Applying sigmoid here would
        # silently change the numbers a caller compares against the reference.
        flat = [float(s[0]) if isinstance(s, list) else float(s) for s in scores]

        results = [
            {"index": i, "relevance_score": score} for i, score in enumerate(flat)
        ]
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        top_n = body.get("top_n")
        if isinstance(top_n, int) and top_n > 0:
            results = results[:top_n]
        if body.get("return_documents"):
            for r in results:
                r["document"] = {"text": documents[r["index"]]}

        return {
            "id": f"rerank-{int(time.time() * 1000)}",
            "model": body.get("model") or served_model_name,
            "results": results,
            "usage": {"total_tokens": sum(len(p) for p in packed)},
        }

    # Cohere's shape, which Jina and vLLM both follow; there is no OpenAI
    # standard for reranking. The three paths are the ones clients try.
    @app.post("/v1/rerank")
    def rerank_v1(body: dict = Body(...)):
        return _rerank(body)

    @app.post("/v2/rerank")
    def rerank_v2(body: dict = Body(...)):
        return _rerank(body)

    @app.post("/rerank")
    def rerank_bare(body: dict = Body(...)):
        return _rerank(body)

    return app
