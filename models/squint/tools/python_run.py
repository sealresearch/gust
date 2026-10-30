"""Runs one sequence through the `sealresearch` Python package, one frame per Model.run, and writes PREFIX.py.jsonl for
tools/compare_api.mjs. One process per sequence: the hosts keep their state in the library.

Usage: GUST_LIB=out/lib/windows-x64/gust.dll python models/squint/tools/python_run.py out/squint-seq/NAME
(add the package to the path: PYTHONPATH=packages/python)
"""
import json
import sys

import sealresearch as sr

prefix = sys.argv[1]
with open(prefix + ".cmd") as f:
    lines = f.read().split("\n")
width, height, frames = map(int, lines[0].split())
with open(prefix + ".gray", "rb") as f:
    gray = f.read()

model = sr.Model.open("squint-1")
out = []
for i in range(frames):
    reset, time = lines[i + 1].split()
    frame = sr.Image(gray[i * width * height:(i + 1) * width * height], width, height, "gray8")
    options = {"time": float(time)}
    if reset == "1":
        options["reset"] = True
    out.append(json.dumps({"i": i, "r": model.run(frame, options=options)}))
model.close()
with open(prefix + ".py.jsonl", "w") as f:
    f.write("\n".join(out) + "\n")
