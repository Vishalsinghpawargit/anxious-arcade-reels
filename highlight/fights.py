"""Find every fight in a BGMI / PUBG Mobile esports match by reading the broadcast frame by frame.

The old way cut a match into a few equal parts and kept the loudest window in each, so a match only
ever offered a handful of candidates and most fights were never looked at at all. This reads the
match densely instead: four frames a second, watching the parts of the screen that only change when
something happens in the game.

The useful trick is the standings panel down the right edge. When a team gets a finish, its number
goes up, its alive-player bars change, and the table re-sorts. The panel is translucent, so the
moving game world shows through it and a plain frame-to-frame difference is never still. Taking the
median picture of each second cancels that moving world, and what survives from one second to the
next is the panel's own text changing. On real broadcast frames a finish shows up as roughly 15-20%
of the panel changing hard, against a still-second floor of about 0.6%.

Measured on BMSD 2026 Day 1: at 118 s a team read 6 finishes / 19 points, at 120 s it read 7 / 20,
and by 126 s it had 8 / 22 and had moved up the table. Both finishes show as clear spikes.

The panel also hides itself at times. That changes almost the whole region at once, so a change
above `PANEL_GONE` is the panel coming or going, not a finish.

Two different overlays live in that corner, though. The standings table only moves when the game
moves it, but the win-probability cards ("WWCD 41.07%") re-calculate every second or two on their
own. Telling them apart by how often they change does not work: a real fight moves the table just
as constantly. So this does not claim to count finishes. It reports how busy the scoreboard is, as
one input among several, and a fight has to be backed by the sound and the movement as well.

Everything here runs locally and nothing is capped: every fight the match had comes out, and Claude
is left to judge them as an editor rather than to find them.
"""

import subprocess

import numpy as np

import ai_clips

DENSE_FPS = 4            # frames read per second of video
W, H = 960, 540          # scan size; small panel text needs to survive the downscale
HARD = 70                # a pixel changing by more than this between seconds is text, not noise
PANEL_GONE = 0.45        # more of the panel than this changing at once means it appeared or hid
BUSY_LIFT = 3.0          # how far above its own floor a panel change has to sit to count at all
QUIET = 0.35             # action score below this is calm
LOUD = 1.00              # a fight has to reach this at least once
GAP = 5                  # a fight ends after this many calm seconds
MERGE = 8                # fights closer together than this are really one fight
SHORTEST = 10            # anything shorter than this is noise
LONGEST = 60             # a longer stretch is trimmed to its best seconds, not kept whole
LEAD = 4                 # seconds kept before the action starts, so a fight is not joined late
TAIL = 5                 # seconds kept after it ends, for the confirmation and the reaction
HOP = 0.025              # seconds per step of the sound analysis, for catching gunshot onsets

# Starting point only: `calibrate` measures the real layout per video.
REGIONS = {
    'panel': (0.11, 0.63, 0.80, 1.00),   # standings / probability cards down the right edge
    'team': (0.03, 0.26, 0.00, 0.145),   # the watched team's four players, top left
    'view': (0.00, 0.82, 0.00, 0.80),    # the game world
    'logo': (0.855, 0.99, 0.00, 0.17),   # the event logo, on screen only during live play
}


CALM = 0.5               # an overlay pixel wobbles less than this share of what the world does
HOLD_COL = 0.30          # share of a column that must be calm for it to be inside the overlay
HOLD_ROW = 0.70          # share of a row that must be calm, measured across the overlay's columns


def box(regions, name):
    """A region as array slices."""
    top, bottom, left, right = regions[name]
    return slice(int(top * H), int(bottom * H)), slice(int(left * W), int(right * W))


def snapshot(src, at, seconds=6):
    """A few seconds of frames, for seeing which parts of the screen hold still."""
    out = subprocess.run([ai_clips.FFMPEG, '-hide_banner', '-loglevel', 'error', '-threads', '0',
                          '-ss', f'{at:.2f}', '-t', str(seconds), '-i', str(src), '-map', '0:v:0',
                          '-vf', f'fps={DENSE_FPS},scale={W}:{H},format=gray', '-f', 'rawvideo', '-'],
                         capture_output=True, creationflags=ai_clips.NO_WINDOW).stdout
    count = len(out) // (W * H)
    return np.frombuffer(out[:count * W * H], np.uint8).reshape(count, H, W)


def run_of(share, lo, hi, enough):
    """The longest stretch between `lo` and `hi` that stays above `enough`."""
    best, run, start = (lo, lo), None, None
    for i in range(lo, hi):
        if share[i] > enough:
            start = i if start is None else start
        elif start is not None:
            run, start = (start, i), None
            best = run if run[1] - run[0] > best[1] - best[0] else best
    if start is not None and hi - start > best[1] - best[0]:
        best = (start, hi)
    return best


def calibrate(src, moments):
    """Find where this broadcast's standings overlay actually sits.

    The layout is not the same from event to event, or even from day to day: BMSD Day 1 put a
    nine-team table in the top half of the right edge, Day 2 a sixteen-team one down almost the
    whole edge. Guessing wrong means reading the wrong strip of screen for the entire scan. So the
    overlay is measured instead: over a few seconds the game world moves and an overlay does not,
    so the pixels that hold still are the overlay.
    """
    wobbles = []
    for at in moments:
        frames = snapshot(src, at)
        if len(frames) >= 4:
            wobbles.append(frames.std(axis=0))
    if not wobbles:
        return dict(REGIONS)
    wobble = np.median(wobbles, axis=0)
    # Judge steadiness against the footage itself rather than a fixed number of grey levels. The
    # overlays are translucent, so an absolute threshold throws most of the table away; measured on
    # BMSD Day 2 the game view wobbles about 35 grey levels and the standings edge 0.7, and half
    # the world's wobble separates them whatever the footage looks like.
    world = float(np.median(wobble[:, :int(0.60 * W)])) or 1.0
    calm = wobble < world * CALM

    # The panel is the block of calm columns against the right edge, and then its own rows.
    columns = calm[int(0.05 * H):int(0.99 * H), :].mean(axis=0)
    left, _ = run_of(columns, int(0.62 * W), W, HOLD_COL)
    if left >= int(0.97 * W):  # nothing found, so keep what is known
        return dict(REGIONS)
    rows = calm[:, left:].mean(axis=1)
    top, bottom = run_of(rows, int(0.05 * H), int(0.99 * H), HOLD_ROW)
    if bottom - top < int(0.15 * H):
        return dict(REGIONS)

    found = dict(REGIONS)
    found['panel'] = (top / H, min(0.99, (bottom + 1) / H), left / W, 1.0)
    found['view'] = (0.0, 0.82, 0.0, left / W)
    return found


def scan(src, start, end, update, label, regions):
    """Read one match densely. Returns a signal per second of the match.

    Skipping B-frames cuts about a third off H.264, and is quietly ignored by VP9, which has none.
    Either way the fps filter fixes the sampling rate, so memory does not depend on the codec.
    """
    span = max(1.0, end - start)
    cmd = [ai_clips.FFMPEG, '-hide_banner', '-loglevel', 'error', '-skip_frame', 'bidir',
           '-ss', f'{start:.2f}', '-t', f'{span:.2f}', '-i', str(src), '-map', '0:v:0',
           '-vf', f'fps={DENSE_FPS},scale={W}:{H},format=gray', '-fps_mode', 'cfr',
           '-threads', '0', '-f', 'rawvideo', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, creationflags=ai_clips.NO_WINDOW)
    panel_box, team_box, view_box, logo_box = (box(regions, n)
                                               for n in ('panel', 'team', 'view', 'logo'))

    held = {'panel': [], 'team': [], 'logo': []}   # this second's frames, until the second is full
    steady = {'panel': [], 'team': [], 'logo': []}  # one median picture per second
    motion, shifting, before, size = [], [], None, W * H
    while len(raw := proc.stdout.read(size)) == size:
        frame = np.frombuffer(raw, np.uint8).reshape(H, W)
        held['panel'].append(frame[panel_box])
        held['team'].append(frame[team_box])
        held['logo'].append(frame[logo_box][::2, ::2])
        shifting.append(0.0 if before is None
                        else float(np.abs(frame[view_box].astype(np.int16) - before[view_box]).mean()))
        before = frame
        if len(held['panel']) == DENSE_FPS:
            for name, frames in held.items():
                steady[name].append(np.median(frames, axis=0))
                frames.clear()
            motion.append(float(np.mean(shifting)))
            shifting.clear()
            if len(motion) % 60 == 0:
                update(label, min(99.0, len(motion) / span * 100))
    proc.stdout.close()
    proc.wait()

    seconds = len(motion)
    if not seconds:
        return None
    return {'seconds': seconds,
            'panel': text_change(np.array(steady['panel'])),
            'team': text_change(np.array(steady['team'])),
            'motion': np.array(motion, np.float32),
            'playing': live_seconds(np.array(steady['logo']))}


def text_change(pictures):
    """How much of a panel's text changed from one second to the next, as a share of its pixels."""
    if len(pictures) < 2:
        return np.zeros(len(pictures), np.float32)
    step = np.abs(np.diff(pictures, axis=0)) > HARD
    return np.r_[0.0, step.mean(axis=(1, 2))].astype(np.float32)


def live_seconds(logos):
    """True where the event logo is on screen, which means live play rather than a replay or an ad."""
    import esports
    typical = np.median(logos, axis=0)
    distance = np.abs(logos.astype(np.float32) - typical).mean(axis=(1, 2))
    return distance < esports.otsu(distance)


def onsets(wav):
    """How sharply the sound jumps, per second. Gunfire is many hard onsets; speech is not."""
    samples = ai_clips.read_audio(wav)
    step = max(1, int(ai_clips.RATE * HOP))
    usable = len(samples) // step * step
    if not usable:
        return np.zeros(0, np.float32)
    power = (samples[:usable].reshape(-1, step) ** 2).mean(axis=1)
    energy = np.log10(power + 1e-8)  # the floor has to sit under real quiet, or every onset vanishes
    rise = np.diff(energy, prepend=energy[:1]).clip(min=0)
    per_second = int(round(1 / HOP))
    keep = len(rise) // per_second * per_second
    return rise[:keep].reshape(-1, per_second).sum(axis=1).astype(np.float32)


def lift(values, window=181):
    """How far a signal sits above its own normal level, so a quiet match is judged on its own terms."""
    values = np.asarray(values, np.float32)
    if len(values) < 5:
        return np.zeros_like(values)
    pad = window // 2
    usual = np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(values, pad, mode='edge'), window), axis=1)[:len(values)]
    spread = float(np.median(np.abs(values - usual))) or 1e-6
    return (values - usual) / (spread * 4)


def span_of(values, start, seconds):
    """The part of a whole-video per-second signal that belongs to this match."""
    piece = np.asarray(values, np.float32)[int(start):int(start) + seconds]
    if len(piece) >= seconds:
        return piece[:seconds]
    return np.pad(piece, (0, seconds - len(piece)), mode='edge' if len(piece) else 'constant')


def score_match(signals, db, fire, start):
    """An action score for every second of the match, and where the finishes were."""
    seconds = signals['seconds']
    panel = signals['panel']
    # A finish moves a small part of the panel a long way. The panel hiding moves nearly all of it.
    # Not a finish count: the self-updating cards move for their own reasons. It is how busy the
    # scoreboard is, which a real fight makes busy too.
    busy = ((lift(panel) > BUSY_LIFT) & (panel < PANEL_GONE) & (panel > 0.02)).astype(np.float32)
    score = (1.3 * busy
             + 1.0 * np.clip(lift(signals['team']), 0, 4)
             + 0.6 * np.clip(lift(signals['motion']), 0, 4)
             + 0.9 * np.clip(lift(span_of(db, start, seconds)), 0, 4)
             + 0.8 * np.clip(lift(span_of(fire, start, seconds)), 0, 4))
    score *= 0.45 + 0.55 * signals['playing']  # action inside a replay or a player cam counts for less
    return score / 4.0, busy


def events(score, busy):
    """Every stretch where the action stays up, as (first second, last second)."""
    found, run, calm = [], None, 0
    for i, hot in enumerate(score > QUIET):
        if hot:
            run = i if run is None else run
            calm = 0
        elif run is not None:
            calm += 1
            if calm >= GAP:
                found.append((run, i - calm))
                run = None
    if run is not None:
        found.append((run, len(score) - 1))

    merged = []
    for a, b in found:
        if merged and a - merged[-1][1] <= MERGE:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged
            if b - a >= SHORTEST and (score[a:b + 1].max() >= LOUD or busy[a:b + 1].sum() >= 3)]


def tighten(score, a, b):
    """The strongest `LONGEST` seconds of an over-long stretch.

    A fight that rolls straight into a rotation and then another skirmish comes back as one long
    run. Keeping it whole is how a highlight turns into the stream with the breaks removed, so the
    best window is kept instead. Length is controlled here rather than by demanding a higher score,
    because a higher bar loses whole fights and that is the failure worth avoiding.
    """
    if b - a <= LONGEST:
        return a, b
    run = np.convolve(score[a:b + 1], np.ones(LONGEST), 'valid')
    best = a + int(np.argmax(run))
    return best, best + LONGEST


def analyse(src, start, end, db, fire, update, label, regions=None):
    """The per-second action score of one match, and how busy its scoreboard was."""
    signals = scan(src, start, end, update, label, regions or REGIONS)
    if not signals:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    score, busy = score_match(signals, db, fire, start)
    return score, busy, signals['playing'].astype(np.float32)


def fights_in(score, busy, playing, start, end):
    """Every fight in one match, in time order. Each is a dict of its times and what was in it."""
    found = []
    for a, b in events(score, busy):
        a, b = tighten(score, a, b)
        found.append({
            'start': float(max(start, start + a - LEAD)),
            'end': float(min(end, start + b + TAIL)),
            'scoreboard': int(busy[a:b + 1].sum()),
            'peak': round(float(score[a:b + 1].max()), 2),
            'heat': round(float(score[a:b + 1].mean()), 2),
            'live': round(float(playing[a:b + 1].mean()), 2) if len(playing) > b else 1.0,
        })
    return found
