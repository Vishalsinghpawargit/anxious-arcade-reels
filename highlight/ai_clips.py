"""AI highlight video and Shorts/Reels from a local video file.

The heavy work runs on this PC for free: reading the audio, the loudness scan that finds the
exciting moments, and speech-to-text (faster-whisper). Claude Haiku 4.5 only reads short
transcript excerpts of the best candidate moments, so even an hours-long video costs a few
thousand tokens. Claude is reached through Claude Code, so it runs on the user's Claude
subscription instead of API credits.
"""

import json
import os
import re
import shutil
import site
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from pydantic import BaseModel

MODEL = 'haiku'  # Claude Haiku 4.5, the lightest model
WHISPER_MODEL = 'large-v3-turbo'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
RATE = 16000                    # sample rate of the analysis audio
FULL_TRANSCRIPT_MAX = 20 * 60   # videos up to this long are transcribed completely
WINDOW = 60                     # seconds per candidate moment of a longer video
FONTS = [('Nirmala UI', 'Nirmala.ttc'), ('Arial', 'arialbd.ttf')]  # Nirmala UI covers Hindi and Latin

SYSTEM = """You edit short-form video. From transcript excerpts of one video, pick the most engaging moments.
- Shorts: self-contained 20-60 s moments with a strong hook in the first 3 seconds. Start and end on sentence boundaries. Shorts must not overlap.
- Highlight video: the best moments in time order, each 8-40 s, adding up to the requested total.
- Only use times inside the given excerpts. Times are seconds from the start of the video.
- "hype" is how much louder than normal a moment is. Loud moments are often the exciting ones.
- Titles: catchy, under 60 characters, in the language of the video. For Hindi use Hinglish in Latin script.
- Description: one or two sentences. Hashtags: 3 to 6, each starting with #."""


class Clip(BaseModel):
    start: float
    end: float
    title: str


class Short(Clip):
    description: str
    hashtags: list[str]


class Plan(BaseModel):
    highlights: list[Clip]
    shorts: list[Short]


def clock(seconds):
    s = int(seconds)
    return f'{s // 3600}:{s // 60 % 60:02}:{s % 60:02}' if s >= 3600 else f'{s // 60}:{s % 60:02}'


def _run(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8',
                            errors='replace', creationflags=NO_WINDOW)
    if result.returncode != 0:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        raise RuntimeError(lines[-1] if lines else 'ffmpeg failed.')
    return result


def probe(path):
    """Length in seconds and whether the file has sound."""
    info = subprocess.run([FFMPEG, '-hide_banner', '-i', str(path)], capture_output=True, text=True,
                          encoding='utf-8', errors='replace', creationflags=NO_WINDOW).stderr
    found = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', info)
    if not found:
        raise RuntimeError('This file is not a video that can be read.')
    h, m, s = found.groups()
    return int(h) * 3600 + int(m) * 60 + float(s), ' Audio:' in info


def read_audio(wav, start=0.0, end=None):
    with wave.open(str(wav)) as w:
        w.setpos(min(int(start * RATE), w.getnframes()))
        count = w.getnframes() - w.tell() if end is None else int((end - start) * RATE)
        data = w.readframes(count)
    return np.frombuffer(data, np.int16).astype(np.float32) / 32768


def loudness(wav):
    """Loudness in dB of every second of audio."""
    levels = []
    with wave.open(str(wav)) as w:
        while chunk := w.readframes(RATE):
            x = np.frombuffer(chunk, np.int16).astype(np.float32)
            levels.append(10 * np.log10(np.mean(x * x) + 1))
    return np.array(levels)


def find_moments(db, count, window):
    """(start, end, hype) of the loudest windows, measured against the loudness around them."""
    typical = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(db, 150, mode='edge'), 301), axis=1)
    rel = db - typical
    scored = sorted(((np.sort(rel[s:s + window])[-10:].mean(), s)
                     for s in range(0, max(1, len(db) - window + 1), 5)), reverse=True)
    picked = []
    for score, s in scored:
        if all(abs(s - p) >= window for _, p in picked):
            picked.append((score, s))
            if len(picked) == count:
                break
    return sorted((s, min(s + window, len(db)), round(float(score), 1)) for score, s in picked)


_whisper_model = None


def _whisper(update):
    global _whisper_model
    if _whisper_model is None:
        # CUDA libraries come from the nvidia-* pip packages, so no separate CUDA install is needed.
        for base in site.getsitepackages():
            for sub in ('nvidia/cublas/bin', 'nvidia/cudnn/bin'):
                if Path(base, sub).is_dir():
                    os.environ['PATH'] = f'{Path(base, sub)}{os.pathsep}{os.environ["PATH"]}'
        from faster_whisper import WhisperModel, download_model
        update('Getting the speech model (the first run downloads about 1.6 GB)', None)
        path = download_model(WHISPER_MODEL)
        try:
            _whisper_model = WhisperModel(path, device='cuda', compute_type='int8_float16')
        except Exception:
            _whisper_model = WhisperModel(path, device='cpu', compute_type='int8')
    return _whisper_model


def transcribe(wav, ranges, update):
    """Speech inside the (start, end) ranges as segments with word timings, in video time."""
    model = _whisper(update)
    total = sum(e - s for s, e in ranges) or 1
    done, spoken, segments = 0.0, {}, []
    for s, e in ranges:
        # Commentary often mixes languages (Hindi and English), so each stretch is detected on its own.
        parts, info = model.transcribe(read_audio(wav, s, e), word_timestamps=True,
                                       vad_filter=True, condition_on_previous_text=False)
        before = len(segments)
        for seg in parts:
            segments.append({
                'start': s + seg.start, 'end': s + seg.end, 'text': seg.text.strip(),
                'words': [(s + w.start, s + w.end, w.word.strip()) for w in seg.words or []],
            })
            update('Turning speech into text', min(100, (done + seg.end) / total * 100))
        if len(segments) > before:
            spoken[info.language] = spoken.get(info.language, 0) + e - s
        done += e - s
    return segments, max(spoken, key=spoken.get) if spoken else None


def plain_schema(model):
    """JSON schema of a pydantic model with nested models written out in place of $ref links,
    which the model follows more reliably."""
    schema = model.model_json_schema()
    defs = schema.pop('$defs', {})

    def resolve(node):
        if isinstance(node, dict):
            if '$ref' in node:
                return resolve(defs[node['$ref'].split('/')[-1]])
            return {k: resolve(v) for k, v in node.items() if not (k == 'title' and isinstance(v, str))}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def claude_cli():
    """Path of the Claude Code command, or None when it is not installed."""
    installed = Path.home() / '.local' / 'bin' / 'claude.exe'
    return shutil.which('claude') or (str(installed) if installed.exists() else None)


def ask_claude(system, prompt, schema, thinking=0):
    """One Claude Code request in print mode, answered as JSON that matches `schema`.

    `prompt` is text, or a list of content blocks when images are included.
    `thinking` caps the tokens Claude may spend thinking before it answers.
    It runs on the Claude Code login, so it uses the Claude subscription, not API credits.
    No tools, settings, plugins or MCP servers are loaded, so only this prompt is sent.
    """
    cli = claude_cli()
    if not cli:
        raise RuntimeError('Claude Code is not installed. Install it and log in with your Claude account.')
    # With an API key in the environment Claude Code would bill API credits instead of the subscription.
    env = {k: v for k, v in os.environ.items() if k not in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN')}
    # Uncapped, Haiku thinks at length: about 25x the output tokens and 15x the time for the same picks.
    env['MAX_THINKING_TOKENS'] = str(thinking)
    cmd = [cli, '-p', '--model', MODEL, '--system-prompt', system, '--tools', '', '--setting-sources', '',
           '--strict-mcp-config', '--no-session-persistence', '--disable-slash-commands', '--json-schema', json.dumps(schema)]
    if isinstance(prompt, str):
        cmd += ['--output-format', 'json']
    else:
        # Images go in as a stream-json user message; the reply is then a stream of JSON events.
        cmd += ['--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose']
        prompt = json.dumps({'type': 'user', 'message': {'role': 'user', 'content': prompt}}) + '\n'
    # Now and then the model ends without a valid structured answer, so that is asked once more.
    for _ in range(2):
        with tempfile.TemporaryDirectory(prefix='yt-ai-claude-') as cwd:
            try:
                result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding='utf-8',
                                        errors='replace', cwd=cwd, env=env, timeout=600, creationflags=NO_WINDOW)
            except subprocess.TimeoutExpired:
                raise RuntimeError('Claude took too long to answer. Try again.')
        try:
            try:
                events = [json.loads(result.stdout)]
            except ValueError:
                events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            out = next(e for e in reversed(events) if isinstance(e, dict) and e.get('type') == 'result')
        except (ValueError, StopIteration):
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise RuntimeError('Claude Code failed: ' + (detail[-1] if detail else f'exit code {result.returncode}'))
        if out.get('is_error'):
            if out.get('subtype') == 'error_max_structured_output_retries':
                continue
            raise RuntimeError(f'Claude Code could not answer: {out.get("result") or out.get("subtype")}')
        if out.get('structured_output') is None:
            try:
                out['structured_output'] = json.loads(out.get('result') or '')
            except ValueError:
                continue
        return out
    raise RuntimeError(f'Claude did not return clip times ({out.get("subtype")}). Try again.')


def plan_clips(name, duration, language, moments, peaks, segments, want_highlight, shorts, target):
    lines = [
        f'Video: "{name}", length {clock(duration)}, speech language: {language or "unknown"}.',
        'Make: ' + (f'a highlight video of {target} s total (stay within 10%)' if want_highlight else 'no highlight video (empty list)')
        + ', and ' + (f'{shorts} shorts.' if shorts else 'no shorts (empty list).'),
    ]
    if peaks:
        lines.append('Loudest moments: ' + ', '.join(f'{s}-{e} s (hype {h:+.1f} dB)' for s, e, h in peaks))
    lines.append('Excerpts, each line starts with its time in seconds:')
    for i, (s, e, hype) in enumerate(moments, 1):
        lines.append(f'\n#{i} {s}-{e} s' + (f', hype {hype:+.1f} dB' if len(moments) > 1 else ''))
        lines += [f'{seg["start"]:.0f} {seg["text"]}' for seg in segments if s <= seg['start'] < e]

    out = ask_claude(SYSTEM, '\n'.join(lines), plain_schema(Plan))
    usage = out.get('usage') or {}
    sent = sum(usage.get(k) or 0 for k in ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'))
    return Plan.model_validate(out['structured_output']), {'input': sent, 'output': usage.get('output_tokens') or 0}


def tidy(plan, duration, want_highlight, shorts):
    """Keep clips inside the video, at most 60 s long, sorted, and without overlaps."""
    def fit(clips):
        kept = []
        for c in sorted(clips, key=lambda c: c.start):
            c.start = max(0.0, c.start)
            c.end = min(duration, c.end, c.start + 60)
            if c.end - c.start >= 2 and (not kept or c.start >= kept[-1].end):
                kept.append(c)
        return kept

    for s in plan.shorts:
        s.hashtags = ['#' + t.strip().lstrip('#').replace(' ', '') for t in s.hashtags if t.strip().lstrip('#')]
    return (fit(plan.highlights) if want_highlight else []), fit(plan.shorts)[:shorts]


_encoder_args = None


def encoder():
    """GPU (NVENC) H.264 encoding when available, otherwise CPU."""
    global _encoder_args
    if _encoder_args is None:
        test = subprocess.run([FFMPEG, '-hide_banner', '-f', 'lavfi', '-i', 'color=s=320x240:d=0.2',
                               '-c:v', 'h264_nvenc', '-f', 'null', '-'], capture_output=True, creationflags=NO_WINDOW)
        _encoder_args = (['-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '23'] if test.returncode == 0
                         else ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21'])
    return _encoder_args


def copy_font(work):
    """Put the caption font next to the subtitles, so ffmpeg does not scan every Windows font."""
    (work / 'fonts').mkdir()
    for family, file in FONTS:
        src = Path(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', file)
        if src.exists():
            shutil.copy(src, work / 'fonts' / file)
            return family
    return 'Arial'


def write_captions(segments, clip, path, font):
    """ASS subtitles of the words spoken in the clip, 3 at a time, placed under the video."""
    words = [w for seg in segments for w in seg['words'] if clip.start <= w[0] < clip.end and w[2]]
    if not words:
        return None
    groups, group = [], []
    for w in words:
        if group and (len(group) == 3 or w[0] - group[-1][1] > 0.6 or w[1] - group[0][0] > 1.8):
            groups.append(group)
            group = []
        group.append(w)
    groups.append(group)

    def t(x):
        x = max(0.0, x - clip.start)
        return f'{int(x // 3600)}:{int(x // 60 % 60):02}:{x % 60:05.2f}'

    events = []
    for i, g in enumerate(groups):
        end = max(g[-1][1], g[0][0] + 0.4)
        if i + 1 < len(groups):
            end = min(end, groups[i + 1][0][0])
        text = ' '.join(w[2] for w in g).replace('{', '').replace('}', '').replace('\\', '')
        events.append(f'Dialogue: 0,{t(g[0][0])},{t(end)},Cap,,0,0,0,,{text}')
    path.write_text(
        '[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 0\n\n'
        '[V4+ Styles]\n'
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, '
        'Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, '
        'MarginR, MarginV, Encoding\n'
        f'Style: Cap,{font},76,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1\n\n'
        '[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
        + '\n'.join(events) + '\n',
        encoding='utf-8',
    )
    return path.name


def render_short(src, clip, out, work, captions):
    """1080x1920: the video in the middle over a blurred copy of itself, captions below it."""
    graph = ('[0:v]split[a][b];'
             '[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:1,scale=1080:1920[bg];'
             '[b]scale=1080:1920:force_original_aspect_ratio=decrease[fg];'
             '[bg][fg]overlay=(W-w)/2:(H-h)/2')
    if captions:
        graph += f',subtitles={captions}:fontsdir=fonts'
    graph += ',format=yuv420p[v]'
    _run([FFMPEG, '-hide_banner', '-y', '-ss', f'{clip.start:.2f}', '-t', f'{clip.end - clip.start:.2f}',
          '-i', str(src), '-filter_complex', graph, '-map', '[v]', '-map', '0:a?', *encoder(),
          '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', str(out)], cwd=work)


def render_highlight(src, clips, out, work):
    """The clips one after another, at most 1080p."""
    for i, c in enumerate(clips):
        _run([FFMPEG, '-hide_banner', '-y', '-ss', f'{c.start:.2f}', '-t', f'{c.end - c.start:.2f}',
              '-i', str(src), '-vf', "scale=-2:'min(1080,ih)',format=yuv420p", *encoder(),
              '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-ac', '2', f'part{i}.mp4'], cwd=work)
    (work / 'parts.txt').write_text(''.join(f"file 'part{i}.mp4'\n" for i in range(len(clips))))
    _run([FFMPEG, '-hide_banner', '-y', '-f', 'concat', '-safe', '0', '-i', 'parts.txt',
          '-c', 'copy', '-movflags', '+faststart', str(out)], cwd=work)


def safe_name(text):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', text).strip(' .')[:60] or 'clip'


def make_clips(src, want_highlight, shorts, update, highlight_length=180):
    """Analyse `src`, then write the highlight video and shorts into a folder next to it.

    `highlight_length` is the wanted length of the highlight video in seconds.
    `update(step, percent)` reports progress; percent is None when unknown.
    Returns (outputs, token usage, output folder).
    """
    src = Path(src)
    duration, has_audio = probe(src)
    if not has_audio:
        raise RuntimeError('This video has no sound, so there is nothing to analyse.')
    target = int(min(highlight_length, duration)) if want_highlight else 0

    with tempfile.TemporaryDirectory(prefix='yt-ai-') as tmp:
        work = Path(tmp)
        update('Reading the audio', None)
        wav = work / 'audio.wav'
        _run([FFMPEG, '-hide_banner', '-y', '-i', str(src), '-vn', '-ac', '1', '-ar', str(RATE),
              '-c:a', 'pcm_s16le', str(wav)])

        update('Finding the exciting moments', None)
        db = loudness(wav)
        if duration <= FULL_TRANSCRIPT_MAX:
            moments, peaks = [(0, int(duration), 0.0)], find_moments(db, 6, 10)
        else:
            # Only the loudest stretches are transcribed and sent to Claude: about 2.5 times the
            # material the highlight needs, so Claude still has a choice.
            count = min(40, max(3 * shorts + 8, -(-target * 5 // (2 * WINDOW))))
            moments, peaks = find_moments(db, count, WINDOW), []
        segments, language = transcribe(wav, [(s, e) for s, e, _ in moments], update)
        if not segments:
            raise RuntimeError('No speech was found in this video, so there is nothing for the AI to read.')

        update('Claude is picking the best moments', None)
        plan, usage = plan_clips(src.stem, duration, language, moments, peaks, segments,
                                 want_highlight, shorts, target)
        highlights, picked = tidy(plan, duration, want_highlight, shorts)

        folder = src.parent / f'{src.stem} - AI clips'
        n = 2
        while folder.exists():
            folder = src.parent / f'{src.stem} - AI clips ({n})'
            n += 1
        folder.mkdir()

        font = copy_font(work)
        total = (1 if highlights else 0) + len(picked)
        outputs = []
        if highlights:
            update('Making the highlight video', 0)
            file = folder / 'Highlights.mp4'
            render_highlight(src, highlights, file, work)
            outputs.append({'kind': 'highlight', 'title': 'Highlights', 'file': str(file),
                            'start': highlights[0].start, 'end': highlights[-1].end,
                            'length': sum(c.end - c.start for c in highlights),
                            'description': ' | '.join(c.title for c in highlights), 'hashtags': []})
        for i, clip in enumerate(picked, 1):
            update(f'Making short {i} of {len(picked)}', len(outputs) / total * 100)
            name = f'Short {i} - {safe_name(clip.title)}'
            captions = write_captions(segments, clip, work / f'short{i}.ass', font)
            render_short(src, clip, folder / f'{name}.mp4', work, captions)
            (folder / f'{name}.txt').write_text(
                f'{clip.title}\n\n{clip.description}\n\n{" ".join(clip.hashtags)}\n', encoding='utf-8')
            outputs.append({'kind': 'short', 'title': clip.title, 'file': str(folder / f'{name}.mp4'),
                            'start': clip.start, 'end': clip.end, 'length': clip.end - clip.start,
                            'description': clip.description, 'hashtags': clip.hashtags})
    return outputs, usage, str(folder)
