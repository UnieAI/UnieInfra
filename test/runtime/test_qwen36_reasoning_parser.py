# Copyright (c) 2026 LightSeek Foundation
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Qwen3.6 launcher must pick a reasoning parser that honours enable_thinking.

The Qwen3.6 chat template exposes an ``enable_thinking`` toggle (default on).
The gateway arms the reasoning parser only when thinking is effectively on, so
the launcher must use the toggle-aware ``qwen3`` parser; ``qwen3_thinking``
starts inside reasoning and would put a thinking-off answer into
``reasoning_content``.
"""

import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "qwen3.6-35b-a3b.sh"


def test_default_reasoning_parser_is_toggle_aware():
    text = SCRIPT.read_text()
    m = re.search(r'^REASONING_PARSER="\$\{REASONING_PARSER:-([^}]*)\}"', text, re.M)
    assert m, "REASONING_PARSER default not found"
    assert m.group(1) == "qwen3"


def test_served_model_name_default_unchanged():
    assert 'SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test}"' in SCRIPT.read_text()
