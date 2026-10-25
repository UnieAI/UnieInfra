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

"""``unieinfer_kernel`` is an import alias of ``tokenspeed_kernel`` (see UNIEINFER.md).

``import unieinfer_kernel.x.y`` returns the very module object ``tokenspeed_kernel.x.y``;
nothing is imported twice. The block below is a copy of the one in
``unieinfer/__init__.py``; keep the copies in step
(test/runtime/test_unieinfer_import_alias.py compares them).
"""

# --- alias block: identical in unieinfer, unieinfer_kernel, unieinfer_scheduler
import importlib
import importlib.abc
import importlib.util
import sys


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec):
        # The real module, imported under its real name. The import system
        # leaves its __spec__/__name__/__file__ alone because they are set.
        return importlib.import_module(self.target)

    def exec_module(self, module) -> None:
        pass

    def get_code(self, fullname):
        # runpy (``python -m alias.x``) runs the target's code as __main__.
        spec = importlib.util.find_spec(self.target)
        return spec.loader.get_code(self.target)

    def is_package(self, fullname) -> bool:
        spec = importlib.util.find_spec(self.target)
        return spec.submodule_search_locations is not None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self, alias: str, target: str) -> None:
        self.alias = alias
        self.target = target

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.alias + "."):
            return None
        real = self.target + fullname[len(self.alias) :]
        real_spec = importlib.util.find_spec(real)
        if real_spec is None:
            return None
        spec = importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(real),
            is_package=real_spec.submodule_search_locations is not None,
        )
        if real_spec.submodule_search_locations is not None:
            spec.submodule_search_locations = list(real_spec.submodule_search_locations)
        return spec


def _install(alias: str, target: str) -> None:
    if not any(isinstance(f, _AliasFinder) and f.alias == alias for f in sys.meta_path):
        # Ahead of PathFinder, which would otherwise load a second copy of
        # each submodule under the alias name from the target's __path__.
        sys.meta_path.insert(0, _AliasFinder(alias, target))
    sys.modules[alias] = importlib.import_module(target)


# --- end alias block

_install("unieinfer_kernel", "tokenspeed_kernel")
