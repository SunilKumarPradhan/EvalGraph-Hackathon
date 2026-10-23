# /// script
# requires-python = ">=3.10"
# dependencies = ["edge-tts>=6.1", "imageio-ffmpeg>=0.5", "numpy>=1.26"]
# ///
"""Add narration, background music and subtitles to a screen recording of the EvalGraph demo.

    uv run scripts/video/make_demo_video.py --video demo.mp4                    # TTS voice + generated music
    uv run scripts/video/make_demo_video.py                                      # no video yet: narration + music only (.m4a)
    uv run scripts/video/make_demo_video.py --video demo.mp4 --voiceover me.wav # your own recorded voice instead of TTS
    uv run scripts/video/make_demo_video.py --video demo.mp4 --music track.mp3  # your own music file
    uv run scripts/video/make_demo_video.py --list-voices

Narration comes from cues.txt (`mm:ss | text`). Each line is spoken by a neural voice (edge-tts: the text is sent to
Microsoft's read-aloud service, nothing else) and placed at its time. Music is ducked under the voice. Output: the video
with the new audio, a soft subtitle track (or burned in with --burn-subs) and a .srt next to it.
The video stream is copied, not re-encoded (unless --burn-subs). Outputs go to scripts/video/out/ by default.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RATE = 44100


@dataclass
class Cue:
    start: float
    text: str
    line: int = 0
    clip: Path | None = None
    dur: float = 0.0


def ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd,
                          capture_output=True, text=True)
    if proc.returncode:
        sys.exit(f"ffmpeg failed:\n{proc.stderr[-2000:]}")


def duration(path: Path) -> float:
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    if not m:
        sys.exit(f"can't read the duration of {path}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def has_audio(path: Path) -> bool:
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    return "Audio:" in out


def parse_cues(path: Path) -> list[Cue]:
    cues = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#>":
            continue
        m = re.fullmatch(r"(\d+):(\d{1,2}(?:\.\d+)?)\s*\|\s*(.+)", line)
        if not m:
            sys.exit(f"{path.name}:{n}: expected 'mm:ss | text', got: {line}")
        cues.append(Cue(int(m[1]) * 60 + float(m[2]), m[3].strip(), n))
    if not cues:
        sys.exit(f"no cues in {path}")
    cues.sort(key=lambda c: c.start)
    return cues


# ---------------------------------------------------------------- voice


async def _tts(cues: list[Cue], voice: str, rate: str, cache: Path) -> None:
    import edge_tts

    for c in cues:
        key = hashlib.sha1(f"{voice}|{rate}|{c.text}".encode()).hexdigest()[:16]
        c.clip = cache / f"{key}.mp3"
        if not c.clip.exists():
            tmp = c.clip.with_suffix(".part")
            await edge_tts.Communicate(c.text, voice, rate=rate).save(str(tmp))
            tmp.replace(c.clip)


def synthesize(cues: list[Cue], voice: str, rate: str, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(_tts(cues, voice, rate, cache))
    except Exception as exc:  # network down, unknown voice, service change
        sys.exit(f"text-to-speech failed ({exc}). Check the internet connection and --voice, "
                 "or record yourself and pass --voiceover.")
    for c in cues:
        c.dur = duration(c.clip)


def timing_report(cues: list[Cue], end: float | None) -> int:
    """Print when each line starts/ends; return how many lines run into the next one."""
    clashes = 0
    print(f"\n{'start':>6} {'end':>6}  {'free':>5}  text")
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else end
        free = None if nxt is None else nxt - (c.start + c.dur)
        flag = ""
        if free is not None and free < 0.2:
            clashes += 1
            flag = f"  << runs {-free + 0.2:.1f} s into the next line: move the next time later or shorten this one"
        print(f"{c.start:6.1f} {c.start + c.dur:6.1f}  {'' if free is None else f'{free:5.1f}'}  "
              f"{c.text[:60]}{'...' if len(c.text) > 60 else ''}{flag}")
    return clashes


def pack(cues: list[Cue], gap: float = 0.5) -> int:
    """Push lines later so none starts before the previous one has finished (+gap). Returns how many moved."""
    moved = 0
    for prev, c in zip(cues, cues[1:]):
        earliest = prev.start + prev.dur + gap
        if c.start < earliest:
            c.start = round(earliest, 1)
            moved += 1
    return moved


def write_timed_cues(src: Path, cues: list[Cue], out: Path) -> Path:
    """The cue sheet with the packed times: use it as the shot list while recording."""
    lines = src.read_text(encoding="utf-8").splitlines()
    for c in cues:
        text = lines[c.line - 1].split("|", 1)[1]
        lines[c.line - 1] = f"{int(c.start // 60):02d}:{c.start % 60:04.1f} |{text}"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------- music


def _env(n: int, attack: int, release: int) -> np.ndarray:
    e = np.ones(n)
    e[:attack] = np.linspace(0, 1, attack)
    e[-release:] *= np.linspace(1, 0, release)
    return e


def generate_music(seconds: float, out: Path, seed: int = 7) -> Path:
    """Original ambient bed (pad chords + bass + soft arpeggio), synthesized here, so there is no licence to clear."""
    rng = np.random.default_rng(seed)
    bpm = 84
    beat = 60 / bpm
    bar = 4 * beat
    # A minor: Am, F, C, G; two bars each
    chords = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]
    total = int((seconds + 1) * RATE)
    left = np.zeros(total)
    right = np.zeros(total)

    def hz(midi: float) -> float:
        return 440.0 * 2 ** ((midi - 69) / 12)

    def tone(freq: float, n: int, harmonics=(1.0, 0.35, 0.12), detune=0.0) -> np.ndarray:
        t = np.arange(n) / RATE
        sig = sum(a * np.sin(2 * np.pi * freq * (k + 1) * (1 + detune) * t) for k, a in enumerate(harmonics))
        return sig

    span = 2 * bar
    pos, i = 0.0, 0
    while pos < seconds + 1:
        notes = chords[i % len(chords)]
        a = int(pos * RATE)
        n = min(int((span + 1.5) * RATE), total - a)  # overlap 1.5 s into the next chord
        if n <= 0:
            break
        env = _env(n, int(1.2 * RATE), int(1.8 * RATE))
        for midi in notes:
            f = hz(midi)
            left[a:a + n] += 0.16 * env * tone(f, n, detune=-0.0025)
            right[a:a + n] += 0.16 * env * tone(f, n, detune=0.0025)
        bass = 0.22 * env * tone(hz(notes[0] - 12), n, harmonics=(1.0, 0.2))
        left[a:a + n] += bass
        right[a:a + n] += bass
        # eighth-note arpeggio, an octave up, quiet, alternating sides
        step = beat / 2
        for k in range(int(span / step)):
            s = a + int(k * step * RATE)
            m = min(int(0.9 * RATE), total - s)
            if m <= 0:
                break
            midi = notes[k % 3] + 12 + (12 if k % 8 == 7 else 0)
            pluck = 0.07 * np.exp(-np.arange(m) / (0.18 * RATE)) * tone(hz(midi), m, harmonics=(1.0, 0.25))
            pan = 0.35 if k % 2 else 0.65
            vel = 0.8 + 0.2 * rng.random()
            left[s:s + m] += vel * pan * pluck
            right[s:s + m] += vel * (1 - pan) * pluck
        pos += span
        i += 1

    stereo = np.stack([left, right], axis=1)
    stereo /= np.max(np.abs(stereo)) + 1e-9
    stereo *= 0.89  # about -1 dBFS
    raw = out.with_suffix(".raw.wav")
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((stereo * 32767).astype("<i2").tobytes())
    # warm it up (low-pass) and add a little room
    run_ffmpeg(["-i", str(raw), "-af", "lowpass=f=3200,aecho=0.8:0.5:120|240:0.25|0.15", str(out)])
    raw.unlink()
    return out


# ---------------------------------------------------------------- subtitles


def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _wrap(text: str, width: int = 44) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


def write_srt(cues: list[Cue], out: Path) -> Path:
    """One subtitle per sentence (or ~90-char chunk), timed in proportion to its length within the spoken clip."""
    blocks = []
    for c in cues:
        parts = []
        for sentence in re.split(r"(?<=[.!?:])\s+", c.text):
            while len(sentence) > 90:
                cut = sentence.rfind(" ", 0, 90)
                cut = cut if cut > 0 else 90
                parts.append(sentence[:cut])
                sentence = sentence[cut:].strip()
            if sentence:
                parts.append(sentence)
        total = sum(len(p) for p in parts) or 1
        t = c.start
        for p in parts:
            d = c.dur * len(p) / total
            blocks.append((t, t + d, p))
            t += d
    out.write_text("\n".join(f"{i}\n{_ts(a)} --> {_ts(b)}\n{_wrap(p)}\n" for i, (a, b, p) in enumerate(blocks, 1)),
                   encoding="utf-8")
    return out


# ---------------------------------------------------------------- mix


def mix(cues: list[Cue], voiceover: Path | None, music: Path | None, length: float, music_db: float,
        video: Path | None, keep_audio_db: float | None, out_audio: Path) -> None:
    """Voice bus (clips placed with adelay, or one voiceover file) + music ducked under it (+ optional original audio)."""
    inputs: list[str] = []
    graph: list[str] = []
    idx = 0
    voice_labels = []
    if voiceover:
        inputs += ["-i", str(voiceover)]
        graph.append(f"[{idx}:a]aresample={RATE},aformat=channel_layouts=stereo[v0]")
        voice_labels.append("[v0]")
        idx += 1
    else:
        for k, c in enumerate(cues):
            inputs += ["-i", str(c.clip)]
            ms = int(c.start * 1000)
            graph.append(f"[{idx}:a]aresample={RATE},aformat=channel_layouts=stereo,adelay={ms}|{ms}[v{k}]")
            voice_labels.append(f"[v{k}]")
            idx += 1
    graph.append(f"{''.join(voice_labels)}amix=inputs={len(voice_labels)}:normalize=0:dropout_transition=0,"
                 f"apad,atrim=0:{length:.3f}[voice]")
    bus = ["[voice]"]
    if music:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        fade_out = max(0.0, length - 3)
        graph.append(f"[{idx}:a]aresample={RATE},aformat=channel_layouts=stereo,atrim=0:{length:.3f},"
                     f"volume={music_db}dB,afade=t=in:d=2,afade=t=out:st={fade_out:.3f}:d=3[music]")
        idx += 1
        graph.append("[voice]asplit=2[vmix][vkey]")
        graph.append("[music][vkey]sidechaincompress=threshold=0.015:ratio=10:attack=30:release=600:makeup=1[duck]")
        bus = ["[vmix]", "[duck]"]
    if video is not None and keep_audio_db is not None and has_audio(video):
        inputs += ["-i", str(video)]
        graph.append(f"[{idx}:a]aresample={RATE},aformat=channel_layouts=stereo,volume={keep_audio_db}dB,"
                     f"apad,atrim=0:{length:.3f}[orig]")
        bus.append("[orig]")
        idx += 1
    if len(bus) > 1:
        graph.append(f"{''.join(bus)}amix=inputs={len(bus)}:normalize=0,alimiter=limit=0.95[out]")
    else:
        graph.append(f"{bus[0]}alimiter=limit=0.95[out]")
    run_ffmpeg([*inputs, "-filter_complex", ";".join(graph), "-map", "[out]", "-c:a", "aac", "-b:a", "192k",
                "-t", f"{length:.3f}", str(out_audio)])


def render(video: Path, audio: Path, srt: Path, out: Path, burn: bool) -> None:
    if burn:
        # run in the .srt folder and pass a bare file name: Windows drive colons break the subtitles filter's syntax
        style = "FontName=Segoe UI,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,MarginV=28"
        run_ffmpeg(["-i", str(video.resolve()), "-i", str(audio.resolve()), "-map", "0:v:0", "-map", "1:a:0",
                    "-vf", f"subtitles={srt.name}:force_style='{style}'", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "-movflags", "+faststart",
                    str(out.resolve())], cwd=srt.parent)
    else:
        run_ffmpeg(["-i", str(video), "-i", str(audio), "-i", str(srt), "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
                    "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
                    "-movflags", "+faststart", str(out)])  # no -shortest: it would stop at the last subtitle


# ---------------------------------------------------------------- main


async def _voices() -> list[dict]:
    import edge_tts

    return await edge_tts.list_voices()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, help="screen recording (.mp4). Omit to render narration + music only")
    p.add_argument("--cues", type=Path, default=HERE / "cues.txt")
    p.add_argument("--out", type=Path, help="output file (default: out/<video>-final.mp4 or out/narration.m4a)")
    p.add_argument("--voice", default="en-IN-PrabhatNeural", help="edge-tts voice (see --list-voices)")
    p.add_argument("--rate", default="+10%", help="speaking rate (default +10%%; +0%% is slower)")
    p.add_argument("--voiceover", type=Path, help="your own narration audio for the whole video (replaces TTS)")
    p.add_argument("--music", default="generate", help="'generate' (default), a music file, or 'none'")
    p.add_argument("--music-db", type=float, default=-12.0, help="music level before ducking (default -12 dB; lower = quieter)")
    p.add_argument("--keep-audio-db", type=float, help="keep the recording's own audio at this level (e.g. -18)")
    p.add_argument("--pack", action="store_true",
                   help="move lines later so none overlaps the previous one (default when there is no --video)")
    p.add_argument("--burn-subs", action="store_true", help="burn subtitles into the picture (re-encodes the video)")
    p.add_argument("--list-voices", action="store_true")
    a = p.parse_args()

    if a.list_voices:
        for v in sorted(asyncio.run(_voices()), key=lambda v: v["ShortName"]):
            if v["Locale"].startswith("en-"):
                print(f"{v['ShortName']:32} {v['Gender']}")
        return

    outdir = HERE / "out"
    outdir.mkdir(exist_ok=True)
    cues = parse_cues(a.cues)
    video_len = duration(a.video) if a.video else None
    if a.video and not a.video.exists():
        sys.exit(f"no such video: {a.video}")

    if a.voiceover:
        vo_len = duration(a.voiceover)
        # subtitles still follow the cue times; each line is assumed to last until the next one starts
        for i, c in enumerate(cues):
            c.dur = max(0.5, (cues[i + 1].start if i + 1 < len(cues) else vo_len) - c.start - 0.3)
        print(f"voiceover: {a.voiceover} ({vo_len:.1f} s); subtitles timed from the cue sheet")
    else:
        print(f"voice: {a.voice} {a.rate} · {len(cues)} lines")
        synthesize(cues, a.voice, a.rate, outdir / "tts-cache")
        if a.pack or not a.video:
            moved = pack(cues)
            timed = write_timed_cues(a.cues, cues, outdir / "cues-timed.txt")
            print(f"packed: {moved} line(s) moved later so nothing overlaps · timed shot list: {timed}")
        clashes = timing_report(cues, video_len)
        if clashes:
            print(f"\nWARNING: {clashes} line(s) overlap the next one. Edit the times in {a.cues.name}, "
                  "or use --pack / --rate +8%.")

    spoken_end = max(c.start + c.dur for c in cues)
    length = video_len if video_len else spoken_end + 3
    if video_len and spoken_end > video_len:
        print(f"WARNING: narration ends at {spoken_end:.1f} s but the video is {video_len:.1f} s: the tail is cut.")

    music = None
    if a.music == "generate":
        music = outdir / "music-generated.wav"
        if not music.exists() or duration(music) < length:
            print(f"generating {length:.0f} s of background music...")
            generate_music(length + 5, music)
    elif a.music != "none":
        music = Path(a.music)
        if not music.exists():
            sys.exit(f"no such music file: {music}")

    stem = a.video.stem if a.video else "narration"
    srt = write_srt(cues, outdir / f"{stem}-final.srt")
    audio = outdir / f"{stem}-mix.m4a"
    mix(cues, a.voiceover, music, length, a.music_db, a.video, a.keep_audio_db, audio)
    if not a.video:
        final = a.out or outdir / "narration.m4a"
        if final != audio:
            shutil.copyfile(audio, final)
        print(f"\nnarration + music: {final} ({length:.1f} s)\nsubtitles: {srt}\n"
              "Play it while you record to keep the pace, then run again with --video.")
        return
    final = a.out or outdir / f"{stem}-final.mp4"
    render(a.video, audio, srt, final, a.burn_subs)
    print(f"\nvideo: {final} ({duration(final):.1f} s)\nsubtitles: {srt} ({'burned in' if a.burn_subs else 'soft track + .srt'})")


if __name__ == "__main__":
    main()
