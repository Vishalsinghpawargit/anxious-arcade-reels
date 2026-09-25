"""Shared pieces: fetching media, measuring action in the sound, and the vertical Reel layout."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'assets'
WORK = ROOT / 'work'
OUT = ROOT / 'out'
FFMPEG = shutil.which('ffmpeg') or 'ffmpeg'
FONT = (ASSETS / 'Anton-Regular.ttf').as_posix().replace(':', '\\:')
LIME = '0xC8F03C'
RATE = 16000     # analysis sample rate
HOP = 0.025      # seconds per step when counting gunshot onsets

for d in (WORK, OUT):
    d.mkdir(exist_ok=True)


def run(cmd):
    print('+', ' '.join(str(c) for c in cmd)[:300], flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def _cookies():
    path = WORK / 'cookies.txt'
    text = os.environ.get('YT_COOKIES', '').strip()
    if text:
        path.write_text(text + '\n')
        return ['--cookies', str(path)]
    return []


def _stop_live_now(data, edge):
    """A live stream downloads from its start up to now instead of following it forever
    (same as the laptop app's stop_live_at)."""
    from urllib.parse import parse_qs, urlparse

    def capped(fragments):
        def generate(ctx):
            for frag in fragments(ctx):
                edge.setdefault('last', frag['fragment_count'])
                if int(parse_qs(urlparse(frag['url']).query)['sq'][0]) >= edge['last']:
                    return
                yield {**frag, 'fragment_count': edge['last']}
        return generate

    for fmt in data.get('formats') or []:
        if fmt.get('is_from_start') and callable(fmt.get('fragments')):
            fmt['fragments'] = capped(fmt['fragments'])


def youtube(url, stem, height=1080, audio_only=False):
    """Download a YouTube (or any yt-dlp supported) link. Returns the file path.
    A stream that is still live is taken from its start up to now."""
    import yt_dlp
    opts = {'noplaylist': True, 'js_runtimes': {'node': {}}, 'concurrent_fragment_downloads': 8,
            'outtmpl': str(WORK / f'{stem}.%(ext)s'), 'live_from_start': True, 'noprogress': True,
            'format': 'bestaudio[ext=m4a]/bestaudio' if audio_only else 'bv*+ba/b',
            'format_sort': [] if audio_only else [f'res:{height}', 'vcodec:h264', 'acodec:aac'],
            'merge_output_format': 'mp4'}
    cookies = _cookies()
    if cookies:
        opts['cookiefile'] = cookies[1]
    with yt_dlp.YoutubeDL(opts) as ydl:
        data = ydl.extract_info(url, download=False)
        if data.get('is_live'):
            print('Stream is live: downloading from its start up to now', flush=True)
            _stop_live_now(data, {})
        data = ydl.process_ie_result(data, download=True)
    return Path(data['requested_downloads'][0]['filepath'])


def fetch(url, stem):
    """Any video link: Google Drive share link, direct file link, or YouTube."""
    if 'drive.google.com' in url:
        import gdown
        out = WORK / f'{stem}.mp4'
        gdown.download(url, str(out), fuzzy=True, quiet=False)
        return out
    if re.search(r'\.(mp4|mov|m4a|mp3|webm)(\?|$)', url, re.I):
        out = WORK / (stem + Path(url.split('?')[0]).suffix)
        run(['curl', '-L', '--fail', '-o', out, url])
        return out
    return youtube(url, stem)


def probe(path):
    info = subprocess.run([FFMPEG, '-hide_banner', '-i', str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r'Duration: (\d+):(\d+):([\d.]+)', info).groups()
    size = re.search(r'Video:.*?(\d{2,5})x(\d{2,5})', info)
    w, hgt = map(int, size.groups()) if size else (0, 0)
    return int(h) * 3600 + int(m) * 60 + float(s), w, hgt, ' Audio:' in info


def read_audio(path):
    raw = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-vn', '-ac', '1', '-ar', str(RATE),
                          '-f', 's16le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32)


def action_score(path):
    """Per-second action: loudness plus sharp sound onsets (gunfire), each normalised, smoothed 4 s."""
    x = read_audio(path)
    n = len(x) // RATE
    sec = x[:n * RATE].reshape(n, RATE)
    loud = 10 * np.log10((sec * sec).mean(axis=1) + 1)
    step = int(RATE * HOP)
    y = x[:len(x) // step * step] / 32768
    energy = np.log10((y.reshape(-1, step) ** 2).mean(axis=1) + 1e-8)
    rise = np.diff(energy, prepend=energy[:1]).clip(min=0)
    per = int(round(1 / HOP))
    onsets = rise[:len(rise) // per * per].reshape(-1, per).sum(axis=1)
    n = min(len(loud), len(onsets))
    z = lambda a: (a[:n] - np.median(a[:n])) / (np.percentile(a[:n], 90) - np.median(a[:n]) + 1e-6)
    return np.convolve(z(loud) + z(onsets), np.ones(4) / 4, 'same'), z(onsets)


def most_replayed(url, fallback_fraction=0.35):
    """Start (s) of the most replayed part of a YouTube video, from its replay heatmap."""
    try:
        out = subprocess.run(['yt-dlp', '--js-runtimes', 'node', *_cookies(), '--skip-download',
                              '--print', '%(heatmap)j', '--print', '%(duration)s', url],
                             capture_output=True, text=True, check=True).stdout.splitlines()
        heat, dur = json.loads(out[0]), float(out[1])
        if heat:
            return max(heat, key=lambda h: h['value'])['start_time']
        return dur * fallback_fraction
    except Exception as e:
        print('heatmap unavailable:', e)
        return None


def text(t, size, color, y, border=4):
    t = t.replace("'", '').replace(':', '\\:')
    return (f"drawtext=fontfile='{FONT}':text='{t}':fontsize={size}:fontcolor={color}:"
            f"x=(w-tw)/2:y={y}:borderw={border}:bordercolor=black")


def title_size(line, big):
    return big if len(line) <= 12 else int(big * 0.75) if len(line) <= 17 else int(big * 0.6)


def reel(clips, top, main, footer, out, captions=None, crop_panel=True):
    """Join (file, start, length) clips into a 1080x1920 Reel: blurred fill, gameplay in the middle,
    a two-line title on top and the channel line at the bottom. Original sound kept."""
    inputs, parts = [], []
    for i, (src, start, length) in enumerate(clips):
        inputs += ['-ss', f'{start:.2f}', '-t', f'{length:.2f}', '-i', str(src)]
        cap = f',{text(captions[i], 70, "white", 1400)}' if captions and captions[i] else ''
        fg = 'crop=iw*0.76:ih:iw*0.04:0,scale=1080:800' if crop_panel else 'scale=1080:-2'
        parts.append(
            f'[{i}:v]setpts=PTS-STARTPTS,fps=30,setsar=1,split[b{i}][f{i}];'
            f'[b{i}]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,'
            f'gblur=sigma=30,eq=brightness=-0.25[g{i}];[f{i}]{fg},setsar=1[h{i}];'
            f'[g{i}][h{i}]overlay=0:(H-h)/2{cap}[v{i}];'
            f'[{i}:a]aresample=48000,afade=t=in:d=0.15,afade=t=out:st={max(0, length - 0.25):.2f}:d=0.25,'
            f'asetpts=PTS-STARTPTS[a{i}]')
    n = len(clips)
    ts, ms = title_size(top, 110), title_size(main, 150)
    graph = (';'.join(parts) + ';' + ''.join(f'[v{i}][a{i}]' for i in range(n)) +
             f'concat=n={n}:v=1:a=1[cv][ca];'
             f'[cv]{text(top, ts, LIME, 250, 5)},{text(main, ms, "white", 370, 6)},'
             f'{text(footer, 50, "white", 1500, 3)},{text("@AnxiousArcade", 62, LIME, 1580, 3)},'
             'setsar=1,format=yuv420p[v];[ca]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]')
    run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', *inputs, '-filter_complex', graph,
         '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'medium',
         '-crf', '21', '-r', '30', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
         '-movflags', '+faststart', out])
    return out


def slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')[:60] or 'reel'
