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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Apply the ``UNIEINFER_*`` environment aliases as ``tokenspeed`` is imported.

The implementation is ``tokenspeed_kernel/env_alias.py``. Importing it the
normal way would run ``tokenspeed_kernel/__init__`` (Triton, a device check),
which ``import tokenspeed`` must not pay for, yet the CLI reads
``TOKENSPEED_*`` variables before the kernel package is ever imported. So the
stdlib-only file is loaded on its own, under a private name.
"""

from __future__ import annotations

import importlib.util
import os


def apply_env_aliases() -> None:
    """Copy each ``UNIEINFER_X`` onto ``TOKENSPEED_X`` (the new name wins)."""
    kernel = importlib.util.find_spec("tokenspeed_kernel")
    if kernel is None or not kernel.submodule_search_locations:
        return
    for directory in kernel.submodule_search_locations:
        path = os.path.join(directory, "env_alias.py")
        if os.path.isfile(path):
            break
    else:
        return
    spec = importlib.util.spec_from_file_location("_tokenspeed_env_alias", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_aliases()
