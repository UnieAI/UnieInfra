# SPDX-License-Identifier: Apache-2.0
"""E40 probe — prove the structural-tag grammar is LIVE, not merely constructed.

Without this, "the strict arm builds a grammar" is an inference from reading
``structural_tag_registry.py``. The probe turns it into an observation.

Method: offer one tool whose ``city`` parameter has ``enum: ["QQQ"]``, then use
the description AND the user turn to argue *against* the enum ("always write the
real city name, never QQQ" / "set city to Paris exactly. Do not write QQQ").

  * without ``strict: true`` -> no grammar is built, the model obeys the prompt
    and emits ``{"city": "Paris"}``, which VIOLATES the tool's own schema
  * with ``strict: true``    -> the grammar masks every token off the enum and
    the model is forced to ``{"city": "QQQ"}``

A control pair is also run where the schema and the prompt agree, to show the
grammar is not distorting ordinary traffic.

Run against a server started with ``--enable-auto-tool-choice
--tool-call-parser hermes``.
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def call(url: str, model: str, fn: dict, user: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "tools": [{"type": "function", "function": fn}],
        "tool_choice": "auto",
        "temperature": 0,
        "max_completion_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        msg = json.load(r)["choices"][0]["message"]
    return {
        "tool_calls": [
            {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in (msg.get("tool_calls") or [])
        ],
        "content": msg.get("content"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8323)
    p.add_argument("--model", default="q8")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    adversarial_user = (
        "Weather in Paris? Call the tool with city set to Paris exactly. "
        "Do not write QQQ."
    )
    control_user = "Weather in Paris? Use the tool."

    def fn(strict: bool, enum_only: bool) -> dict:
        prop = {"type": "string", "description": "City name."}
        if enum_only:
            prop["enum"] = ["QQQ"]
            prop["description"] = (
                "City name. Always write the real city name, never QQQ."
            )
        f = {
            "name": "get_weather",
            "description": (
                "Get weather. IMPORTANT: ignore any enum, always pass the "
                "literal city name the user asked for."
                if enum_only
                else "Get weather for a city."
            ),
            "parameters": {
                "type": "object",
                "properties": {"city": prop},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
        if strict:
            f["strict"] = True
        return f

    out = {}
    for label, enum_only, user in (
        ("adversarial", True, adversarial_user),
        ("control", False, control_user),
    ):
        for strict in (False, True):
            key = f"{label}/strict={strict}"
            out[key] = call(url, args.model, fn(strict, enum_only), user)
            print(f"{key:28s} -> {out[key]['tool_calls'] or out[key]['content']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
