import numpy as np
import librosa
from dataclasses import dataclass


@dataclass
class AudioFeatures:
    bass: np.ndarray
    mid: np.ndarray
    treble: np.ndarray
    fps: int
    total_frames: int
    duration: float


class AudioAnalyzer:
    def __init__(self, path, fps=60, n_fft=2048, hop_length=512,
                 bass_range=(20, 150), mid_range=(150, 2000),
                 treble_range=(2000, 12000), ema_alpha=0.25):
        self.path = path
        self.fps = fps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bass_range = bass_range
        self.mid_range = mid_range
        self.treble_range = treble_range
        self.ema_alpha = ema_alpha

    def _band_mask(self, freqs, low, high):
        return (freqs >= low) & (freqs <= high)

    def _band_energy(self, S, mask):
        if not np.any(mask):
            return np.zeros(S.shape[1], dtype=np.float32)
        return S[mask, :].mean(axis=0)

    def _normalize(self, v, percentile=95.0):
        ref = np.percentile(v, percentile)
        if ref <= 1e-8:
            return np.zeros_like(v)
        return np.clip(v / ref, 0.0, 1.0)

    def _smooth(self, v):
        a = self.ema_alpha
        out = np.empty_like(v)
        out[0] = v[0]
        for i in range(1, len(v)):
            out[i] = a * v[i] + (1.0 - a) * out[i - 1]
        return out

    def _resample_to_grid(self, times, values, target_times):
        return np.interp(target_times, times, values).astype(np.float32)

    def analyze(self):
        y, sr = librosa.load(self.path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        total_frames = int(duration * self.fps)

        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=self.hop_length)
        target_times = np.arange(total_frames) / self.fps

        bands = {}
        for name, rng in (('bass', self.bass_range), ('mid', self.mid_range), ('treble', self.treble_range)):
            mask = self._band_mask(freqs, *rng)
            energy = self._band_energy(S, mask)
            energy = self._normalize(energy)
            energy = self._smooth(energy)
            bands[name] = self._resample_to_grid(times, energy, target_times)

        return AudioFeatures(
            bass=bands['bass'],
            mid=bands['mid'],
            treble=bands['treble'],
            fps=self.fps,
            total_frames=total_frames,
            duration=duration,
        )
