"""Read who the broadcast is following, at every keyframe of a BGMI broadcast.

At 1080p two HUD regions are stacked and read with OCR:
  top-left player list  (names like TAGxHarsh, iQOORNTXNinjA)
  followed player's nameplate above the health bar (team full name + player name)
Run directly as a worker: python scan.py <video> <out.json> <start> <end>
"""
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

W, H = 480, 300
GRAPH = ('scale=1920:1080,split[l][p];[l]crop=300:200:0:80,pad=480:200[l2];[p]crop=480:100:560:850[p2];'
         '[l2][p2]vstack,showinfo')


def worker(src, out, a, b):
    from rapidocr_onnxruntime import RapidOCR
    import shutil
    cmd = [shutil.which('ffmpeg') or 'ffmpeg', '-hide_banner', '-skip_frame', 'nokey', '-ss', str(a),
           '-t', str(b - a), '-i', src, '-copyts', '-vf', GRAPH, '-fps_mode', 'passthrough', '-an',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    times = []

    def read_err():
        for line in p.stderr:
            m = re.search(rb'pts_time:([\d.]+)', line)
            if m:
                times.append(float(m.group(1)))

    threading.Thread(target=read_err, daemon=True).start()
    ocr = RapidOCR()
    rows = []
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        res, _ = ocr(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
        top = [r[1] for r in (res or []) if r[0][0][1] < 200 and re.search('x', r[1], re.I)]
        plate = [r[1] for r in (res or []) if r[0][0][1] >= 200]
        while len(times) <= len(rows):
            threading.Event().wait(0.01)
        rows.append((times[len(rows)], top, plate))
    json.dump(rows, open(out, 'w'))


def scan(src, duration, out_json):
    """Split the broadcast across CPU cores, one OCR worker each."""
    cores = max(1, (os.cpu_count() or 2))
    edges = np.linspace(0, duration, cores + 1)
    parts = [f'{out_json}.{i}' for i in range(cores)]
    with ThreadPoolExecutor(cores) as pool:
        list(pool.map(lambda i: subprocess.run(
            [sys.executable, __file__, str(src), parts[i], str(edges[i]), str(edges[i + 1])], check=True),
            range(cores)))
    rows = sorted(r for p in parts for r in json.load(open(p)))
    json.dump(rows, open(out_json, 'w'))
    for p in parts:
        Path(p).unlink()
    return rows


if __name__ == '__main__':
    worker(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
