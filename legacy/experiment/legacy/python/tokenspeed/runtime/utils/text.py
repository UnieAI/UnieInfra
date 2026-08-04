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


def _is_cjk_char(codepoint: int):
    if (
        (0x4E00 <= codepoint <= 0x9FFF)
        or (0x3400 <= codepoint <= 0x4DBF)
        or (0x20000 <= codepoint <= 0x2A6DF)
        or (0x2A700 <= codepoint <= 0x2B73F)
        or (0x2B740 <= codepoint <= 0x2B81F)
        or (0x2B820 <= codepoint <= 0x2CEAF)
        or (0xF900 <= codepoint <= 0xFAFF)
        or (0x2F800 <= codepoint <= 0x2FA1F)
    ):
        return True

    return False


def find_printable_text(text: str):
    if text.endswith("\n"):
        return text
    if len(text) > 0 and _is_cjk_char(ord(text[-1])):
        return text
    if len(text) > 1 and _is_cjk_char(ord(text[-2])):
        return text[:-1]
    return text[: text.rfind(" ") + 1]
