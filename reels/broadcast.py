"""Reels from full BGMI broadcasts.

team mode: one fight montage per team, from every broadcast given.
day mode:  one Reel of the day's hardest fights, any team.

python -m reels.broadcast team --urls "URL1 URL2" --teams "TAG=Team Apex Gaming; SOUL=iQOO SOUL" --max 180
python -m reels.broadcast day  --urls "URL" --top "GLIMPSES OF" --main "DAY 3" --length 60
"""
import argparse
import json
import re

import numpy as np

from . import common, scan

STOP = {'iqoo', 'team', 'esports', 'esport', 'gaming', 'official', 'the', 'and', 'club'}
IQ = r'(?:[i1l]?[qo0d]{2,4}\s?x?)?'
QUIET, GAP, LEAD, TAIL, SHORTEST, BUSY = 0.45, 5, 2, 2, 6, 0.2


def team_patterns(spec):
    """'TAG=Team Apex Gaming; SOUL=iQOO SOUL' -> {TAG: regex}. The tag is matched at the start of
    player names (after an optional iQOO-style sponsor prefix); distinctive words of the full
    name are matched in the nameplate."""
    teams = {}
    for part in re.split(r'[;\n]', spec):
        if not part.strip():
            continue
        tag, _, full = part.partition('=')
        tag = tag.strip().upper()
        tag_rx = ''.join('[O0]' if c == 'O' else re.escape(c) for c in tag)
        words = [w for w in re.findall(r'[a-z0-9]+', full.lower()) if w not in STOP and len(w) >= 4]
        alts = [rf'^{IQ}{tag_rx}'] + [re.escape(w) for w in words]
        teams[tag] = (full.strip() or tag, re.compile('|'.join(alts), re.I))
    return teams


def owners(rows, teams, length):
    """Which team is followed at each second, from consecutive keyframe reads."""
    def at(row):
        texts = [t.replace(' ', '') for t in row[1] + row[2]]
        return {k for k, (_, rx) in teams.items() if any(rx.search(t) for t in texts)}
    own = np.full(length, '', dtype=object)
    for a, b in zip(rows, rows[1:]):
        ta, tb = at(a), at(b)
        if len(ta) == 1 and ta == tb and b[0] - a[0] <= 12:
            own[int(a[0]):int(b[0]) + 1] = next(iter(ta))
        elif len(ta) == 1:
            own[int(a[0]):int(a[0]) + 4] = next(iter(ta))
    return own


def followed_any(rows, length):
    """Seconds where the broadcast follows a player (nameplate or player list on screen): live play."""
    live = np.zeros(length, bool)
    for a, b in zip(rows, rows[1:]):
        if (a[1] or len(a[2]) >= 2) and b[0] - a[0] <= 12:
            live[int(a[0]):int(b[0]) + 1] = True
    return live


def fights(mask, score, busy):
    hot = np.flatnonzero(mask & (score > QUIET))
    segs = []
    for i in hot:
        if segs and i - segs[-1][1] <= GAP:
            segs[-1][1] = i
        else:
            segs.append([i, i])
    out = []
    for a, b in segs:
        a, b = max(0, a - LEAD), min(len(score), b + TAIL)
        if b - a >= SHORTEST and busy[a:b].mean() > BUSY:  # gunfire, not just engine noise or talk
            out.append((int(a), int(b), float(score[a:b].mean())))
    return out


def cut(src, a, b, dest):
    common.run([common.FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-ss', a, '-t', b - a, '-i', src,
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-c:a', 'aac', '-b:a', '192k', dest])


def analyse(urls):
    """Download, scan and score each broadcast in turn. The source is deleted after its candidate
    clips are cut, so disk use stays at one broadcast."""
    for n, url in enumerate(urls):
        src = common.youtube(url, f'b{n}')
        dur = int(common.probe(src)[0])
        score, busy = common.action_score(src)
        rows = scan.scan(src, dur, str(common.WORK / f'b{n}.json'))
        length = min(len(score), dur)
        yield n, src, rows, score[:length], busy[:length], length


def team_mode(args):
    teams = team_patterns(args.teams)
    found = {k: [] for k in teams}
    clipdir = common.WORK / 'clips'
    clipdir.mkdir(exist_ok=True)
    for n, src, rows, score, busy, length in analyse(args.urls.split()):
        own = owners(rows, teams, length)
        for tag in teams:
            for a, b, s in fights(own == tag, score, busy):
                dest = clipdir / f'{tag}_{n}_{a}.mp4'
                cut(src, a, b, dest)
                found[tag].append((s, n, a, str(dest), b - a))
        src.unlink()
    for tag, (full, _) in teams.items():
        chosen, used = [], 0
        for f in sorted(found[tag], reverse=True):
            if used + f[4] <= args.max:
                chosen.append(f)
                used += f[4]
        chosen.sort(key=lambda f: (f[1], f[2]))
        print(tag, len(found[tag]), 'fights found,', len(chosen), 'used,', used, 's')
        if not chosen:
            continue
        common.reel([(f[3], 0, f[4]) for f in chosen], full.upper(), args.main or 'FIGHT MONTAGE',
                    args.footer, common.OUT / f'{common.slug(tag)}_Fights.mp4')


def day_mode(args):
    picks = []
    for n, src, rows, score, busy, length in analyse(args.urls.split()):
        live = followed_any(rows, length)
        cand = sorted(((score[i:i + 12].mean(), i) for i in range(0, length - 12)
                       if live[i:i + 12].all() and busy[i:i + 12].mean() > BUSY), reverse=True)
        mine = []
        for s, i in cand:
            if all(abs(i - j) >= 20 for _, j in mine):
                mine.append((s, i))
            if len(mine) * 12 >= args.length * 2:
                break
        for s, i in mine:
            dest = common.WORK / f'day_{n}_{i}.mp4'
            cut(src, i - 1, i + 12, dest)
            picks.append((float(s), n, i, str(dest)))
        src.unlink()
    picks = sorted(picks, reverse=True)[:max(1, args.length // 13)]
    picks.sort(key=lambda p: (p[1], p[2]))
    common.reel([(p[3], 0, 13) for p in picks], args.top, args.main, args.footer,
                common.OUT / f'{common.slug(args.top + " " + args.main)}.mp4')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['team', 'day'])
    ap.add_argument('--urls', required=True)
    ap.add_argument('--teams', default='')
    ap.add_argument('--max', type=int, default=180)
    ap.add_argument('--length', type=int, default=60)
    ap.add_argument('--top', default='GLIMPSES OF')
    ap.add_argument('--main', default='')
    ap.add_argument('--footer', default='BMSD 2026')
    args = ap.parse_args()
    team_mode(args) if args.mode == 'team' else day_mode(args)
