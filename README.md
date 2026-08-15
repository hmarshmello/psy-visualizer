# psy-visualizer

Turn any MP3 into a fully synced, pre-rendered psychedelic video — driven entirely by math, not AI. Audio is analyzed offline into frequency-band data, then every frame is raymarched on the GPU from a fractal SDF whose parameters are literally the audio's bass/mid/treble energy.

No keyframing, no templates, no diffusion model. Same audio in → same visual out, every time.

## How it works

```
MP3 ──▶ audio_engine.py ──▶ per-frame bass/mid/treble arrays
                                        │
shaders/*.{vert,frag} ──▶ shader_engine.py ──▶ raw RGB frame (headless GPU render)
                                        │
                              video_pipeline.py ──▶ FFmpeg (stdin pipe) ──▶ output.mp4
```

1. **Audio analysis** (`audio_engine.py`) — Librosa computes an STFT, isolates bass (20–150Hz), mid (150–2000Hz), and treble (2000–12000Hz) energy per frame, normalizes it (95th-percentile clip), smooths it with an EMA (α=0.25), and resamples it onto a uniform grid at your target FPS so frame count never drifts from audio duration.
2. **Rendering** (`shader_engine.py`) — A headless ModernGL context (no window, no display) renders a GLSL fragment shader per frame: raymarching through a Mandelbox-style fractal SDF, kaleidoscope-folded in screen space, domain-warped, lit, and colored with a cosine palette. Bass drives camera shake/strobe/FOV, mid drives rotation/fractal scale, treble drives color phase/dolly. The FBO and texture are allocated once and reused every frame.
3. **Encoding** (`video_pipeline.py`) — Rendered frames are piped directly into FFmpeg's stdin as raw RGB — no frames ever touch disk — and muxed with the original audio into the final MP4.

## Requirements

- Python 3.9+
- FFmpeg on `PATH`
- A GPU with OpenGL 3.3+ drivers (headless/EGL supported on Linux; native context on Windows/macOS)

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py input.mp3 output.mp4 --width 1920 --height 1080 --fps 60
```

| Flag | Default | Description |
|---|---|---|
| `--width` | 1920 | Output width |
| `--height` | 1080 | Output height |
| `--fps` | 60 | Frame rate (also the audio analysis grid rate) |

For a fast first check, render small and short before committing to a full 1080p60 pass:

```bash
python main.py input.mp3 preview.mp4 --width 640 --height 360
```

## Sanity-checking your environment

```bash
python -c "import moderngl; ctx = moderngl.create_standalone_context(); print(ctx.info['GL_RENDERER'])"
```

If this fails, you're missing GPU drivers or a headless GL backend (on Linux: `sudo apt install libgl1-mesa-dev libegl1-mesa-dev`).

## Project structure

```
psy_visualizer/
├── main.py              CLI entrypoint / orchestrator
├── audio_engine.py       STFT-based bass/mid/treble extraction
├── shader_engine.py      Headless ModernGL renderer
├── video_pipeline.py     FFmpeg stdin-pipe encoder
├── shaders/
│   ├── fullscreen.vert
│   └── psychedelic.frag  Raymarched Mandelbox + kaleidoscope + domain warp
└── requirements.txt
```

## Notes

- Output is not streamable/playable until FFmpeg finishes writing the MP4's index (`moov` atom) — this is normal for any MP4 encode, not a bug.
- `FFmpegEncoder` caps `-threads` (default 8) and `rc-lookahead=20` to bound peak encoder memory on high-core-count machines; tune via the `threads` constructor arg.

## License

MIT (or your preferred license — update this section).
