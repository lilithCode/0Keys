import numpy as np
import sounddevice as sd
from scipy.signal import chirp
import matplotlib.pyplot as plt

sd.default.device = 'pipewire'

FS = 48000
DURATION = 0.02
F_START = 18000
F_END = 20000

def generate_chirp(fs=FS, duration=DURATION, f_start=F_START, f_end=F_END):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    window = np.hanning(len(signal))
    return signal * window

def record_audio(duration_sec, fs=FS):
    """
    Records from the default input device for duration_sec seconds.
    Returns a 1D numpy array of samples (mono).
    """
    print(f"Recording for {duration_sec}s...")
    recording = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  # blocks until recording finishes
    return recording.flatten()  # sd.rec returns shape (N,1); flatten to (N,)

def compute_fft(signal, fs=FS):
    """
    Computes the frequency spectrum of a time-domain signal.
    Returns (frequencies, magnitudes).
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)          # complex numbers: magnitude + phase per frequency bin
    freqs = np.fft.fftfreq(n, d=1/fs)         # maps each FFT output index to its actual Hz value

    # FFT output is symmetric for real-valued input (mirror image above Nyquist) —
    # we only care about the first half (0 to Nyquist)
    half = n // 2
    freqs = freqs[:half]
    magnitudes = np.abs(fft_result[:half])    # np.abs() of a complex number = its magnitude
    return freqs, magnitudes

def plot_spectrum(freqs, magnitudes, title="Frequency Spectrum"):
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitudes)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.axvline(F_START, color='green', linestyle='--', alpha=0.5, label=f'{F_START/1000}kHz')
    plt.axvline(F_END, color='red', linestyle='--', alpha=0.5, label=f'{F_END/1000}kHz')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    chirp_signal = generate_chirp()

    # Play and record simultaneously isn't set up yet (that's Milestone 6+),
    # so for now: play the chirp, THEN separately record room audio,
    # just to prove recording + FFT works end to end.
    duration_sec = 5.0
    recorded = record_audio(duration_sec)

    freqs, mags = compute_fft(recorded, FS)
    plot_spectrum(freqs, mags, title="FFT of recorded audio")