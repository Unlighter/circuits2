"""Build an executed .ipynb (nbformat 4.5) from a percent-format source file.

Cells are exec'd in one shared namespace; stdout is captured as a stream output and
every open matplotlib figure is captured as a base64 PNG display_data output.
No jupyter / nbformat needed.

usage: python3 build_notebook.py source.txt out.ipynb
"""
import base64
import contextlib
import io
import json
import sys
import time
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 90


def parse(path):
    cells, kind, buf = [], None, []
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, s) for k, s in cells if s.strip()]


def src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def build(source, out):
    cells, ns, n = [], {"__name__": "__main__"}, 0
    for kind, text in parse(source):
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                          "metadata": {}, "source": src(text)})
            continue
        n += 1
        t0 = time.time()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(text, f"<cell {n}>", "exec"), ns)
        outputs = []
        if stdout.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout",
                            "text": src(stdout.getvalue())})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            png = base64.b64encode(buf.getvalue()).decode("ascii")
            outputs.append({"output_type": "display_data",
                            "data": {"image/png": png, "text/plain": ["<Figure>"]},
                            "metadata": {}})
        cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8],
                      "execution_count": n, "metadata": {},
                      "outputs": outputs, "source": src(text)})
        print(f"  cell {n:2d}: {time.time() - t0:6.2f}s  "
              f"{len(outputs)} output(s)", file=sys.stderr)
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}},
        "nbformat": 4, "nbformat_minor": 5}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
