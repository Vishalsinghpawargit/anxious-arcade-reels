"""Full-day match-by-match highlight, the same one the laptop app makes.

python -m reels.full_day --url URL
Release files are capped at 2 GB, so a bigger highlight is split into parts without re-encoding.
"""
import argparse
import shutil
import sys

from . import common

sys.path.insert(0, str(common.ROOT / 'highlight'))
import esports  # noqa: E402

LIMIT = 1.8e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    args = ap.parse_args()
    src = common.youtube(args.url, 'day')
    _, usage, folder = esports.make_match_highlight(src, 0, lambda step, p: print(f'{step} {p or ""}', flush=True))
    print('Claude usage:', usage)
    folder = common.Path(folder)
    shutil.copy(folder / 'Chapters.txt', common.OUT / 'Chapters.txt')
    video = folder / 'Match highlights.mp4'
    size, dur = video.stat().st_size, common.probe(video)[0]
    if size <= LIMIT:
        shutil.move(video, common.OUT / 'Match_highlights.mp4')
        return
    part = int(dur * LIMIT / size)
    common.run([common.FFMPEG, '-hide_banner', '-loglevel', 'error', '-i', video, '-map', '0', '-c', 'copy',
                '-f', 'segment', '-segment_time', part, '-reset_timestamps', '1',
                common.OUT / 'Match_highlights_part%d.mp4'])
    print(f'split into parts of about {part // 60} min; Chapters.txt times are for the whole video')


if __name__ == '__main__':
    main()
