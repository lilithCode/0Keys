import numpy as np
import sounddevice as sd
from scipy.signal import chirp, spectrogram
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
    print(f"Recording for {duration_sec}s...")
    recording = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    return recording.flatten()

def compute_spectrogram(signal, fs=FS, window_ms=20, overlap_pct=75):
 
    nperseg = int(fs * window_ms / 1000)          # samples per chunk
    noverlap = int(nperseg * overlap_pct / 100)     # samples of overlap between chunks

    freqs, times, Sxx = spectrogram(
        signal,
        fs=fs,
        window='hann',      
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum'
    )
    return freqs, times, Sxx

def plot_spectrogram(freqs, times, Sxx, title="Spectrogram"):
    plt.figure(figsize=(10, 5))
    # Sxx is power; convert to dB scale so quiet and loud details are both visible
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # +1e-10 avoids log(0) errors on silent bins
    plt.pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='viridis')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.colorbar(label="Magnitude (dB)")
    plt.axhline(F_START, color='white', linestyle='--', alpha=0.4)
    plt.axhline(F_END, color='white', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    duration_sec = 2.0
    recorded = record_audio(duration_sec)

    freqs, times, Sxx = compute_spectrogram(recorded, FS)
    plot_spectrogram(freqs, times, Sxx, title="Spectrogram of recorded audio")