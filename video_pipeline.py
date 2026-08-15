import subprocess
import threading
from collections import deque


class FFmpegEncoder:
    def __init__(self, width, height, fps, audio_path, output_path,
                 crf=18, preset='medium', audio_bitrate='320k', threads=8):
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_path = audio_path
        self.output_path = output_path
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', f'{width}x{height}',
            '-framerate', str(fps),
            '-i', '-',
            '-i', audio_path,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', preset,
            '-crf', str(crf),
            '-threads', str(threads),
            '-x264-params', 'rc-lookahead=20',
            '-c:a', 'aac',
            '-b:a', audio_bitrate,
            '-shortest',
            output_path,
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._stderr_tail = deque(maxlen=40)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        for line in iter(self.proc.stderr.readline, b''):
            self._stderr_tail.append(line.decode(errors='ignore'))

    def write(self, frame_bytes):
        self.proc.stdin.write(frame_bytes)

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        finally:
            self.proc.wait()
            self._stderr_thread.join(timeout=2)
            if self.proc.returncode != 0:
                err = ''.join(self._stderr_tail)
                raise RuntimeError(f'ffmpeg failed: {err}')
