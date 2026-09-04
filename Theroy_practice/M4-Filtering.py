import numpy as np
import sounddevice as sd
from scipy.signal import chirp, spectrogram, butter, sosfiltfilt
import matplotlib.pyplot as plt

sd.default.device = 'pipewire'

FS = 48000
DURATION = 0.02
F_START = 18000
F_END = 20000

FILTER_LOW = 17000
FILTER_HIGH = 21000
FILTER_ORDER = 6

def generate_chirp(fs=FS, duration=DURATION, f_start=F_START, f_end=F_END):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    window = np.hanning(len(signal))
    return signal * window

def record_audio(duration_sec, fs=FS):
    print(f"Recording for {duration_sec}s...")
    recording = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    return recording.flatten()

def design_bandpass(low, high, fs, order=FILTER_ORDER):
   
    nyquist = fs / 2
    low_norm = low / nyquist    
    high_norm = high / nyquist
    sos = butter(order, [low_norm, high_norm], btype='band', output='sos')
    return sos

def apply_filter(signal, sos):
  
    return sosfiltfilt(sos, signal)

def compute_spectrogram(signal, fs=FS, window_ms=20, overlap_pct=75):
    nperseg = int(fs * window_ms / 1000)
    noverlap = int(nperseg * overlap_pct / 100)
    freqs, times, Sxx = spectrogram(signal, fs=fs, window='hann',
                                     nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    return freqs, times, Sxx

def plot_comparison(raw, filtered, fs=FS):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sig, title in [(axes[0], raw, "Before filtering"),
                             (axes[1], filtered, "After bandpass filtering")]:
        freqs, times, Sxx = compute_spectrogram(sig, fs)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        pcm = ax.pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.axhline(FILTER_LOW, color='white', linestyle='--', alpha=0.4)
        ax.axhline(FILTER_HIGH, color='white', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    duration_sec = 5.0
    recorded = record_audio(duration_sec)

    sos = design_bandpass(FILTER_LOW, FILTER_HIGH, FS)
    filtered = apply_filter(recorded, sos)

    plot_comparison(recorded, filtered)