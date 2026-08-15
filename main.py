import argparse
import os
import sys

from audio_engine import AudioAnalyzer
from shader_engine import ShaderRenderer
from video_pipeline import FFmpegEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input_mp3')
    p.add_argument('output_mp4')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--fps', type=int, default=60)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vert_path = os.path.join(base_dir, 'shaders', 'fullscreen.vert')
    frag_path = os.path.join(base_dir, 'shaders', 'psychedelic.frag')

    analyzer = AudioAnalyzer(args.input_mp3, fps=args.fps)
    features = analyzer.analyze()

    renderer = ShaderRenderer(args.width, args.height, vert_path, frag_path)
    encoder = FFmpegEncoder(args.width, args.height, args.fps, args.input_mp3, args.output_mp4)

    try:
        for i in range(features.total_frames):
            uniforms = {
                'u_time': float(i / args.fps),
                'u_bass': float(features.bass[i]),
                'u_mid': float(features.mid[i]),
                'u_treble': float(features.treble[i]),
            }
            frame_bytes = renderer.render_frame(uniforms)
            encoder.write(frame_bytes)
            if i % args.fps == 0:
                print(f'{i // args.fps}s / {features.duration:.1f}s', file=sys.stderr)
    finally:
        encoder.close()
        renderer.release()


if __name__ == '__main__':
    main()
