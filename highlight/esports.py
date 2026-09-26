"""Match-by-match highlight video of a BGMI / PUBG Mobile esports broadcast.

Free local analysis finds the structure of the day:
- Gameplay frames share the broadcast's HUD, the team list along the right edge. Frames whose
  right edge looks like that are gameplay, and long runs of gameplay are the matches.
- After a match the broadcast shows graphics: the chicken dinner team stats, top players, the
  points table. They move less than camera shots, so the stillest moments after each match go on
  one sheet per match and Claude says which is which.
- Every match opens on the broadcast's own flight path map and the drop that follows it, so the
  video shows a match beginning rather than cutting from one match's table into the next one's
  first fight.
- Inside a match every fight is found by reading it frame by frame (see fights.py), not by taking
  the loudest few stretches, which used to leave most of a match unexamined. The last minute is
  always kept whole as the final fight.
Claude is then given every candidate with its numbers and the casters' words, and acts as the
editor: what earns its place, where each clip starts and ends, what it is called. Nothing is
dropped to hit a length.
"""

import base64
import hashlib
import io
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

import ai_clips
import fights as fight_finder

W, H = 160, 90             # size of the frames used for the scan
SAMPLE = 0.5               # frames a second taken to find the day's structure
MIN_MATCH = 10 * 60        # shorter gameplay runs are replays or recaps
MAX_CUTAWAY = 180          # breaks inside a match (player cams, replays) up to this long are merged
FIGHT_WINDOW = 45          # seconds per candidate fight
FINAL_FIGHT = 60           # seconds of gameplay kept before a match ends
FLIGHT_AT = 45             # seconds before the start where the flight path map is on screen
FLIGHT_BACK = 100          # never open a match earlier than this before it starts
DROP_SHOW = 70             # seconds after the start kept, so the players are seen landing
WWCD_SHOW = 12             # seconds of the chicken dinner screen at most
MVP_SHOW = 20              # seconds of the match's best-players screen at most
TABLE_SHOW = 35            # seconds of the points table at most; it often has two pages
MAX_CLIP = 55              # no single fight clip runs longer than this; trimmed from 70 to
                           # make room for the match openings, which add about 14 minutes
AFTER_MATCH = 15 * 60      # the chicken dinner and points table graphics come within this time
CANDIDATES = 10            # graphic candidates shown to Claude per match
CUT = 25                   # picture change above this between keyframes is a cut to another shot
TILE = (480, 270)

SYSTEM = """You are a highlight editor with twenty years of cutting competitive shooter broadcasts. Today you are cutting one day of a BGMI (PUBG Mobile) esports event into a match-by-match highlight.

You are not being asked to find the action. Every candidate below was already found by reading the broadcast frame by frame. Your job is the editorial one: decide what earns its place, where each clip starts and ends, and what to call it.

For each match you get:
- A sheet of frames. Tile S is shortly before the match, usually the flight path map. Strips S1-S3 under the tiles are the bottom-left corner of the screen before the match at full size, where the flight path map names the match and map, for example "GROUP A - MATCH 1 - RONDO". Tiles A, B, C... are graphics shown after the match.
- A numbered list of candidate fights. For each: its seconds, how busy the scoreboard was (finishes and knocks move it, but the win-probability cards move on their own, so treat it as a hint), how loud it was and how hard the gunfire was, and "live", the share of it that was live play rather than a replay or a player camera.
- What the casters said inside each candidate, each line starting with its time in seconds.

How to cut:
- Keep every real fight: a missed fight is the worst outcome. But a clip is the fight itself, not the rotation around it. Most run 20 to 60 seconds, and each candidate has already been trimmed to its strongest minute, so stay close to the times you are given instead of widening them. Drop a candidate only when the commentary and the numbers agree that nothing happened in it: a rotation, a loot run, a quiet drive, the casters talking over an empty map. A candidate whose "live" is low is usually a replay or a player camera, so keep it only when the commentary shows it matters, and never keep a replay of a fight you are already keeping live.
- Where two or three candidates are really one engagement, return one clip that covers all of it rather than three that cut it up.
- Start a few seconds before the first shot, so the viewer arrives at the setup and not the aftermath. End after the confirmation and the reaction, not on the last bullet. Never end in the middle of an exchange.
- Order the clips by time. They must not overlap.
- Vary the cut: a run of identical mid-game skirmishes is dull, so when several are interchangeable keep the best of them and let the weaker one go. Third parties, clutches, team wipes, close-quarter fights and the end-zone circles always stay.
- Put a funny or emotional beat in when the commentary shows one. It earns its place as the breather between fights.

For each clip give a "kind": "wipe" when a team is finished off, "clutch" when one player holds against odds, "endzone" for the last circles, "thirdparty" when a third team crashes in, "funny" for a comic or emotional beat, and "fight" for anything else.

Also for each match:
- chicken_dinner: the tile with the winning team's chicken dinner (WWCD) screen or team stats.
- mvp: the tile showing the match's best players: individual players with their photos and numbers such as finishes, damage or survival time, for example TOP 5 PLAYERS or an MVP card for one player. This is a different tile from the points table, and both are always kept in the video.
- points_table: the tile with the points standings, a table of TEAMS with their points (columns such as FINISHES, POS. PTS, TOTAL). A table of individual players is the mvp tile, not this one. Answer "" when no tile fits; ads, desk shots and schedules never fit.
- final_standings: for the LAST match only, the tiles with the overall standings at the end of the day, one per group (for example one GROUP A and one GROUP B tile), each showing the top of its table (rank 1 onwards), in group order. Read the group and the ranks in the close-ups of the last match. Empty when there are none.
- map: the map name as written in strips S1-S3 or said in the commentary (Erangel, Miramar, Sanhok, Vikendi, Rondo, Livik). Do not guess from the terrain; answer "" when it is not written or said.
- title: short and catchy, in the language of the commentary. For Hindi use Hinglish in Latin script."""


class Fight(BaseModel):
    start: float
    end: float
    title: str
    kind: Literal['wipe', 'clutch', 'endzone', 'thirdparty', 'funny', 'fight']


Tile = Literal['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']  # one per candidate (CANDIDATES = 10)


class MatchPick(BaseModel):
    match: int
    map: str
    chicken_dinner: Tile | Literal['']
    mvp: Tile | Literal['']
    points_table: Tile | Literal['']
    fights: list[Fight]


class DayPick(BaseModel):
    matches: list[MatchPick]
    final_standings: list[Tile]


CACHE = Path(__file__).with_name('.analysis')


def cached(src, part, make):
    """Run `make()` once per video and keep the arrays it returns.

    Reading a six-hour broadcast takes over an hour, and almost all of it is the same however the
    cut is then assembled. Keeping it means changing a length or a threshold costs minutes instead
    of starting again. A cache that cannot be read or written is simply ignored.
    """
    try:
        stat = Path(src).stat()
        name = f'{Path(src).resolve()}|{stat.st_size}|{stat.st_mtime_ns}'
        path = CACHE / f'{hashlib.sha1(name.encode()).hexdigest()[:16]}-{part}.npz'
        if path.exists():
            with np.load(path) as kept:
                return {key: kept[key] for key in kept.files}
    except Exception:
        path = None
    made = make()
    try:
        if path is not None:
            CACHE.mkdir(exist_ok=True)
            np.savez(path, **made)
    except Exception:
        pass
    return made


def scan_frames(src, duration, update):
    """Small grey copies of the video, one every couple of seconds, and their times.

    These used to be the keyframes, asked for with `-skip_frame nokey`. Some decoders ignore that
    flag and hand back every frame instead: VP9 does, which is what YouTube serves above 1080p, and
    this recording's container marks every frame as a keyframe so `-discard nokey` does not help
    either. On a six-hour 1440p60 file that is 1.3 million frames and about 19 GB of memory rather
    than the 11,000 frames expected. Asking for a fixed rate works whatever the codec turns out to be.
    """
    cmd = [ai_clips.FFMPEG, '-hide_banner', '-threads', '0', '-i', str(src), '-map', '0:v:0',
           '-vf', f'fps={SAMPLE},scale={W}:{H},format=gray,showinfo',
           '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=ai_clips.NO_WINDOW)
    times = []

    def read_times():
        for line in proc.stderr:
            if b'showinfo' in line and b'pts_time:' in line:
                times.append(float(line.split(b'pts_time:')[1].split()[0]))

    reader = threading.Thread(target=read_times, daemon=True)
    reader.start()
    data = bytearray()
    while chunk := proc.stdout.read(W * H * 500):
        data += chunk
        if times:
            update('Finding the matches', min(100, times[-1] / duration * 100))
    proc.wait()
    reader.join()
    count = min(len(data) // (W * H), len(times))
    frames = np.frombuffer(bytes(data[:count * W * H]), np.uint8).reshape(count, H, W)
    return np.array(times[:count]), frames


def otsu(values):
    """The threshold that best splits `values` into a low and a high group."""
    hist, edges = np.histogram(values, bins=128)
    centers = (edges[:-1] + edges[1:]) / 2
    best, threshold = -1.0, edges[1]
    for i in range(1, len(hist)):
        low, high = hist[:i], hist[i:]
        if not low.sum() or not high.sum():
            continue
        m_low = (low * centers[:i]).sum() / low.sum()
        m_high = (high * centers[i:]).sum() / high.sum()
        between = low.sum() * high.sum() * (m_low - m_high) ** 2
        if between > best:
            best, threshold = between, edges[i]
    return threshold


def _step(x):
    """Mean change of each frame region from the previous keyframe."""
    return np.r_[np.inf, np.abs(np.diff(x, axis=0)).mean(axis=tuple(range(1, x.ndim)))]


# Where each broadcast keeps its event logo during live play, in 160x90 scan frames (rows, columns).
LOGOS = {
    # BMSD: the event logo and "QUALIFIERS  WEEK 1" in the bottom-left HUD box. Group and match number
    # next to them change, so they are left out. On Day 1 of BMSD 2026 this found 82-91% of every
    # match's gameplay frames and none in the breaks.
    'BMSD': ((77, 89), (0, 27)),
    # PMGO: the "PUBG MOBILE GLOBAL OPEN" logo above the bottom-left match box. On PMGO S2 EECA Finals
    # Day 1 the BMSD box marked 95% of the day as gameplay; this one found the 6 matches of 23-28 min.
    'PMGO': ((64, 75), (3, 25)),
}


def _is_game(frames, rows, cols):
    logo = frames[:, rows[0]:rows[1], cols[0]:cols[1]].astype(np.float32)
    center = frames[:, 15:75:2, 30:125:2].astype(np.float32)
    # In gameplay the HUD holds still while the game view moves. Desk shots and countdowns have a
    # still center, ads change everywhere. The median of these seed frames is the typical HUD.
    center_step = _step(center)
    seed = (_step(logo) < np.percentile(_step(logo)[1:], 50)) & (center_step > np.median(center_step[1:]))
    distance = np.abs(logo - np.median(logo[seed], axis=0)).mean(axis=(1, 2))
    return drop_lone(distance < otsu(distance))


def classify(frames, times=None):
    """Per sampled frame: is it gameplay, and how much the picture changed since the one before.

    Each known logo position is tried. The one kept finds the most matches of a real match length;
    the wrong one either finds nothing or runs the whole day together as one match."""
    change = _step(frames[:, ::3, ::3].astype(np.float32))
    best, best_count = None, -1
    for rows, cols in LOGOS.values():
        is_game = _is_game(frames, rows, cols)
        count = len([m for m in find_matches(times, is_game) if m[1] - m[0] <= 50 * 60]) if times is not None else 0
        if count > best_count:
            best, best_count = is_game, count
    return best, change


def drop_lone(is_game):
    """Ignore a lone gameplay frame with no gameplay neighbour, so one misread frame cannot extend a match."""
    before = np.r_[False, is_game[:-1]]
    after = np.r_[is_game[1:], False]
    return is_game & (before | after)


def find_matches(times, is_game):
    """(start, end) of every long gameplay run, with short breaks inside a match merged."""
    runs, start, last = [], None, None
    for t, game in zip(times, is_game):
        if not game:
            continue
        if start is None:
            start = t
        elif t - last > MAX_CUTAWAY:
            runs.append((start, last))
            start = t
        last = t
    if start is not None:
        runs.append((start, last))
    return [(s, e) for s, e in runs if e - s >= MIN_MATCH]


def shot(times, change, is_game, i):
    """(start, end) of the shot around keyframe `i`: until a cut or gameplay on either side."""
    a = i
    while a > 0 and change[a] < CUT and not is_game[a - 1]:
        a -= 1
    b = i
    while b + 1 < len(times) and change[b + 1] < CUT and not is_game[b + 1]:
        b += 1
    return float(times[a]), float(times[b])


def opening(times, change, is_game, start):
    """(start, end) of a match's opening: the flight path map through to the players landing.

    Cutting from one match's points table straight into the next match's first fight gives no
    sense that a new match has begun. The broadcast always shows the flight path map first, which
    names the group, the match and the map, then the plane and the drop. All of that is one
    continuous run of footage, so it is kept as a single part: back to the start of whichever shot
    is on screen `FLIGHT_AT` seconds before the match, and forward far enough to see players land.
    """
    near = int(np.argmin(np.abs(times - (start - FLIGHT_AT))))
    begins, _ = shot(times, change, is_game, near)
    return max(start - FLIGHT_BACK, begins), start + DROP_SHOW


def graphic_candidates(times, change, is_game, lo, hi):
    """(time, shot start, shot end) of the stillest non-gameplay keyframes between `lo` and `hi`,
    which is where graphics are shown. One per shot, so a long desk shot does not fill the list."""
    picked = []
    for i in np.argsort(change):
        if lo < times[i] < hi and not is_game[i] and all(not p[1] <= times[i] <= p[2] for p in picked):
            picked.append((float(times[i]), *shot(times, change, is_game, i)))
            if len(picked) == CANDIDATES:
                break
    return sorted(picked)


def frame_at(src, t, vf=f'scale={TILE[0]}:{TILE[1]}'):
    out = subprocess.run([ai_clips.FFMPEG, '-hide_banner', '-loglevel', 'error', '-ss', f'{t:.2f}', '-i', str(src),
                          '-frames:v', '1', '-vf', vf, '-f', 'image2pipe', '-vcodec', 'png', '-'],
                         capture_output=True, creationflags=ai_clips.NO_WINDOW).stdout
    return Image.open(io.BytesIO(out)).convert('RGB') if out else Image.new('RGB', TILE)


def match_sheet(src, start, graphics):
    """JPEG (base64): tile S before the match, tiles A, B, C... for the graphic candidates after it, and
    strips S1-S3, the bottom-left corner at full size before the match, where the map name is written."""
    tiles = [('S', frame_at(src, start - 40))] + [(chr(65 + i), frame_at(src, g[0])) for i, g in enumerate(graphics)]
    strips = [(f'S{i}', frame_at(src, start - 20 - 30 * (i - 1), 'crop=iw*0.45:ih*0.12:0:ih*0.84,scale=864:-2'))
              for i in (1, 2, 3)]
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (TILE[0] * cols, TILE[1] * rows + sum(s.height for _, s in strips)))
    font = ImageFont.truetype(os.environ.get('SHEET_FONT', 'arialbd.ttf'), 40)
    draw = ImageDraw.Draw(sheet)
    places = [(TILE[0] * (i % cols), TILE[1] * (i // cols)) for i in range(len(tiles))]
    y = TILE[1] * rows
    for _, strip in strips:
        places.append((0, y))
        y += strip.height
    for (label, img), (x, y) in zip(tiles + strips, places):
        sheet.paste(img, (x, y))
        width = 52 if len(label) == 1 else 76
        draw.rectangle([x, y, x + width, y + 50], fill='black')
        draw.text((x + 10, y + 3), label, fill='yellow', font=font)
    buf = io.BytesIO()
    sheet.save(buf, 'JPEG', quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def closeup_sheet(src, graphics):
    """JPEG (base64) of the top-left corner of each graphic candidate at full size, where a table's
    title, group and first ranks are readable."""
    crops = [(chr(65 + i), frame_at(src, g[0], 'crop=iw*0.5:ih*0.45:0:0,scale=720:-2')) for i, g in enumerate(graphics)]
    cols, (w, h) = 2, crops[0][1].size
    sheet = Image.new('RGB', (w * cols, h * ((len(crops) + cols - 1) // cols)))
    font = ImageFont.truetype(os.environ.get('SHEET_FONT', 'arialbd.ttf'), 40)
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(crops):
        x, y = w * (i % cols), h * (i // cols)
        sheet.paste(img, (x, y))
        draw.rectangle([x, y, x + 52, y + 50], fill='black')
        draw.text((x + 10, y + 3), label, fill='yellow', font=font)
    buf = io.BytesIO()
    sheet.save(buf, 'JPEG', quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def ask_for_picks(src, plans, segments, language):
    total = sum(len(plan['fights']) for plan in plans)
    content = [{'type': 'text', 'text': f'Commentary language: {language or "unknown"}. '
                                        f'{total} candidate fights were found across {len(plans)} matches. '
                                        f'Keep every one that is a real fight; there is no length limit.'}]
    for i, plan in enumerate(plans, 1):
        start, end = plan['start'], plan['end']
        content.append({'type': 'text', 'text': f'\nMatch {i}: gameplay {start:.0f}-{end:.0f} s. Frames:'})
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                    'data': match_sheet(src, start, plan['graphics'])}})
        if i == len(plans) and plan['graphics']:
            # The end-of-day standings of each group look alike at tile size; the close-ups show the group.
            content.append({'type': 'text', 'text': 'Last match: close-ups of the top-left corner of tiles A, B, C..., '
                                                    'where a table title, its group and its first ranks are readable:'})
            content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                        'data': closeup_sheet(src, plan['graphics'])}})
        lines = []
        for j, f in enumerate(plan['fights'], 1):
            lines.append(f"#{j} {f['start']:.0f}-{f['end']:.0f} s ({f['end'] - f['start']:.0f}s), "
                         f"scoreboard {f['scoreboard']}, loudest {f['peak']}, "
                         f"live {f['live'] * 100:.0f}%")
            lines += [f'{seg["start"]:.0f} {seg["text"]}'
                      for seg in segments if f['start'] <= seg['start'] < f['end']]
        content.append({'type': 'text', 'text': 'Excerpts, each line starts with its time in seconds:\n' + '\n'.join(lines)})
    # Telling the graphics apart is harder than picking times, so this one request may think a little.
    out = ai_clips.ask_claude(SYSTEM, content, ai_clips.plain_schema(DayPick), thinking=2000)
    usage = out.get('usage') or {}
    sent = sum(usage.get(k) or 0 for k in ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'))
    return DayPick.model_validate(out['structured_output']), {'input': sent, 'output': usage.get('output_tokens') or 0}


def graphic(plan, letter, longest, lead=None):
    """(start, end) of the graphic on tile `letter`, at most `longest` seconds.

    It starts at its shot's start, or `lead` seconds before the tile's frame when given: graphics
    that follow each other without a cut (Group A then Group B standings) share one shot.
    """
    index = ord((letter.strip().upper() or '@')[0]) - 65
    if 0 <= index < len(plan['graphics']):
        t, start, end = plan['graphics'][index]
        if lead is not None:
            start = max(start, t - lead)
        return start, max(start + 3, min(end, start + longest))
    return None


def render(src, parts, out, work, update):
    """Encode each (start, end) part with short audio fades, then join them without re-encoding."""
    names = []
    for i, (s, e) in enumerate(parts):
        update(f'Making part {i + 1} of {len(parts)}', i / len(parts) * 100)
        d = e - s
        name = f'part{i:03}.mp4'
        ai_clips._run([ai_clips.FFMPEG, '-hide_banner', '-y', '-ss', f'{s:.2f}', '-t', f'{d:.2f}', '-i', str(src),
                       '-vf', "scale=-2:'min(1080,ih)',format=yuv420p",
                       '-af', f'afade=t=in:d=0.2,afade=t=out:st={max(0, d - 0.3):.2f}:d=0.3',
                       *ai_clips.encoder(), '-maxrate', '9M', '-bufsize', '18M',
                       '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-ac', '2', name], cwd=work)
        names.append(name)
    update('Joining the parts', None)
    (work / 'parts.txt').write_text(''.join(f"file '{n}'\n" for n in names))
    ai_clips._run([ai_clips.FFMPEG, '-hide_banner', '-y', '-f', 'concat', '-safe', '0', '-i', 'parts.txt',
                   '-c', 'copy', '-movflags', '+faststart', str(out)], cwd=work)


def make_match_highlight(src, target, update):
    """Write a match-by-match highlight next to `src`, keeping every fight of every match.

    `target` is only a floor here. The broadcast is read frame by frame and nothing is dropped to
    hit a length, so a busy day makes a long video.

    Returns (outputs, token usage, output folder), like ai_clips.make_clips.
    """
    src = Path(src)
    duration, has_audio = ai_clips.probe(src)
    if not has_audio:
        raise RuntimeError('This video has no sound, so there is nothing to analyse.')

    update('Finding the matches', 0)
    looked = cached(src, 'frames',
                    lambda: dict(zip(('times', 'frames'), scan_frames(src, duration, update))))
    times, frames = looked['times'], looked['frames']
    is_game, change = classify(frames, times)
    # A match still being played when the video ends has no chicken dinner or points table yet.
    matches = [m for m in find_matches(times, is_game) if duration - m[1] > 120]
    if not matches:
        raise RuntimeError('No finished matches were found in this video.')

    with tempfile.TemporaryDirectory(prefix='yt-ai-') as tmp:
        work = Path(tmp)
        update('Reading the audio', None)
        wav = work / 'audio.wav'
        ai_clips._run([ai_clips.FFMPEG, '-hide_banner', '-y', '-i', str(src), '-vn', '-ac', '1', '-ar', str(ai_clips.RATE),
                       '-c:a', 'pcm_s16le', str(wav)])
        db = ai_clips.loudness(wav)
        fire = fight_finder.onsets(wav)

        # Every event lays its HUD out differently, so measure this one instead of assuming.
        update("Learning this broadcast's layout", None)
        regions = fight_finder.calibrate(src, [(s + e) / 2 for s, e in matches[:3]])

        plans = []
        for i, (start, end) in enumerate(matches):
            # Every match is read frame by frame, so no fight can go unseen. The last minute is left
            # out because it is always kept whole as the final fight.
            read = cached(src, f'match{i}', lambda a=start, b=end, n=i: dict(zip(
                ('score', 'busy', 'playing'),
                fight_finder.analyse(src, a, b - FINAL_FIGHT, db, fire, update,
                                     f'Watching match {n + 1} of {len(matches)} frame by frame',
                                     regions))))
            found = fight_finder.fights_in(read['score'], read['busy'], read['playing'],
                                           start, end - FINAL_FIGHT)
            # The day's last standings can come a while after the last match, so it gets the rest of the video.
            limit = min(matches[i + 1][0], end + AFTER_MATCH) if i + 1 < len(matches) else duration
            plans.append({'start': start, 'end': end, 'fights': found,
                          'graphics': graphic_candidates(times, change, is_game, end + 10, limit)})

        ranges = sorted([(f['start'], f['end']) for p in plans for f in p['fights']]
                        + [(p['end'] - FINAL_FIGHT, p['end']) for p in plans])
        segments, language = ai_clips.transcribe(wav, ranges, update)

        update('Claude is choosing what makes the cut', None)
        day, usage = ask_for_picks(src, plans, segments, language)
        picks = {m.match: m for m in day.matches}

        parts, chapters, position = [], [], 0.0
        for i, plan in enumerate(plans, 1):
            pick = picks.get(i)
            name = f'Match {i}' + (f' - {pick.map.strip().title()}' if pick and pick.map.strip() else '')
            chapters.append(f'{ai_clips.clock(position)} {name}')
            # The flight path and the drop come first, so a new match announces itself.
            began = opening(times, change, is_game, plan['start'])
            kept = []
            for f in sorted(pick.fights if pick else [], key=lambda f: f.start):
                # A pick running into the final fight is cut short rather than dropped. Nothing else
                # is trimmed: a long highlight is wanted, a missing fight is not.
                # Nothing may start before the drop is over, or a fight would run over the opening.
                s, e = max(began[1], f.start), min(plan['end'] - FINAL_FIGHT, f.end, f.start + MAX_CLIP)
                if e - s >= 8 and (not kept or s >= kept[-1][1]):
                    kept.append((s, e))
            if not kept:  # Claude skipped the match, so fall back to everything that was detected
                kept = [(f['start'], f['end']) for f in plan['fights'] if f['start'] >= began[1]]
            match_parts = [began] + sorted(kept) + [(plan['end'] - FINAL_FIGHT, plan['end'] + 3)]
            wwcd = graphic(plan, pick.chicken_dinner, WWCD_SHOW) if pick else None
            match_parts.append(wwcd or (plan['end'] + 3, plan['end'] + 3 + WWCD_SHOW))
            # The match's best players are always shown, so this is never dropped for length.
            mvp = graphic(plan, pick.mvp, MVP_SHOW, lead=3) if pick else None
            if mvp and mvp != wwcd:
                match_parts.append(mvp)
            # A points table often has two pages; its tile may show the second, so start a little before it.
            tables = [graphic(plan, pick.points_table, TABLE_SHOW, lead=10)] if pick else []
            final = [graphic(plan, letter, TABLE_SHOW, lead=4) for letter in day.final_standings] if i == len(plans) else []
            if len(final) == 1 and final[0]:
                # One overall table for the whole day (PMGO) runs over two pages in one shot, and the
                # tile Claude sees may be page 2 (ranks 9-16). Start from the shot's start, up to 40 s
                # earlier, and allow both pages. On PMGO S2 EECA Finals Day 1 page 1 came 38 s before.
                final = [graphic(plan, day.final_standings[0], 2 * TABLE_SHOW + 5, lead=40)]
            if any(final):
                # After the last match: every group's overall standings instead of a single table.
                tables = final
            parts += match_parts
            position += sum(e - s for s, e in match_parts)
            if any(final):
                chapters.append(f'{ai_clips.clock(position)} Overall standings')
            for table in tables:
                if table and table not in (wwcd, mvp) and table not in parts:
                    parts.append(table)
                    position += table[1] - table[0]

        folder = src.parent / f'{src.stem} - AI clips'
        n = 2
        while folder.exists():
            folder = src.parent / f'{src.stem} - AI clips ({n})'
            n += 1
        folder.mkdir()
        file = folder / 'Match highlights.mp4'
        render(src, parts, file, work, update)
        (folder / 'Chapters.txt').write_text('\n'.join(chapters) + '\n', encoding='utf-8')

    return ([{'kind': 'highlight', 'title': 'Match highlights', 'file': str(file), 'start': parts[0][0],
              'end': parts[-1][1], 'length': position, 'description': '\n'.join(chapters), 'hashtags': [],
              'chapters': True}],
            usage, str(folder))
