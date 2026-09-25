"""Quick edits on a single video.

endscreen: promo + Anxious Arcade end screen, with the Mortals hook as music.
songhook:  a video with the most replayed part of a song laid over it.

python -m reels.media endscreen --video URL
python -m reels.media songhook --video URL --song URL [--start 95] [--keep-voice]
"""
import argparse

from . import common

MORTALS_HOOK = 129.5   # most replayed part of Warriyo - Mortals
END_LEN = 20.0
UNDER_VOICE = 0.3      # music volume while the video's own sound is playing


def has_real_sound(path):
    import numpy as np
    x = common.read_audio(path)
    return len(x) > 0 and 20 * np.log10(np.sqrt((x / 32768) ** 2).mean() + 1e-9) > -45


def music_graph(music_in, total, dur, voice, keep):
    fade = f'atrim=0:{total},asetpts=PTS-STARTPTS,afade=t=in:d=0.3,afade=t=out:st={total - 2}:d=2'
    if keep:
        return (f'[{music_in}:a]{fade},volume=\'if(lt(t,{dur}),{UNDER_VOICE},1)\':eval=frame[m];'
                f'[0:a]aresample=48000,apad[p];[p][m]amix=inputs=2:duration=shortest:normalize=0[a]')
    return f'[{music_in}:a]{fade}[a]'


def endscreen(args):
    src = common.fetch(args.video, 'promo')
    dur, w, h, audio = common.probe(src)
    w, h = w - w % 2, h - h % 2
    total = dur + END_LEN
    keep = audio and has_real_sound(src)
    start = max(0.0, min(MORTALS_HOOK, 230 - total))
    fit = (f'scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:'
           f'color=0x0e0c2a,setsar=1,fps=30,format=yuv420p')
    graph = (f'[0:v]scale={w}:{h},setsar=1,fps=30,format=yuv420p[v0];[1:v]{fit}[v1];'
             f'[v0][v1]concat=n=2:v=1:a=0[v];' + music_graph(2, total, dur, True, keep))
    out = common.OUT / f'{src.stem}_with_endscreen.mp4'
    common.run([common.FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-i', src,
                '-i', common.ASSETS / 'endscreen.mp4', '-stream_loop', '-1', '-ss', start,
                '-i', common.ASSETS / 'mortals.m4a', '-filter_complex', graph, '-map', '[v]', '-map', '[a]',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-c:a', 'aac', '-b:a', '192k',
                '-t', total, '-movflags', '+faststart', out])


def songhook(args):
    src = common.fetch(args.video, 'video')
    song = common.youtube(args.song, 'song', audio_only=True)
    dur, _, _, audio = common.probe(src)
    start = args.start if args.start is not None else (common.most_replayed(args.song) or 0)
    keep = args.keep_voice and audio and has_real_sound(src)
    graph = music_graph(1, dur, dur, False, keep) if not keep else \
        music_graph(1, dur, dur, True, True).replace(f'if(lt(t,{dur}),{UNDER_VOICE},1)', str(UNDER_VOICE))
    out = common.OUT / f'{src.stem}_with_song.mp4'
    common.run([common.FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-i', src,
                '-stream_loop', '-1', '-ss', start, '-i', song, '-filter_complex', graph,
                '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                '-t', dur, '-movflags', '+faststart', out])
    print(f'song hook starts at {start:.1f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['endscreen', 'songhook'])
    ap.add_argument('--video', required=True)
    ap.add_argument('--song', default='')
    ap.add_argument('--start', type=float, default=None)
    ap.add_argument('--keep-voice', action='store_true')
    args = ap.parse_args()
    endscreen(args) if args.mode == 'endscreen' else songhook(args)
