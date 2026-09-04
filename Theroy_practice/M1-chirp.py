import numpy as np
import sounddevice as sd
from scipy.signal import chirp

FS = 48000
DURATION = 0.2
F_START = 18000     
F_END = 20000  

#For human hearing use this 2500 Hz to 5000 Hz range
# F_START = 1000
# F_END = 3000
sd.default.device = 'pipewire'

def generate_chirp(fs=FS, duration=DURATION, f_start=F_START, f_end=F_END):
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    return signal

def apply_envelope(signal):
 
    window = np.hanning(len(signal))
    return signal * window

def play_chirp(signal, fs=FS):
    sd.play(signal, fs)
    sd.wait()  # blocks until playback finishes

if __name__ == "__main__":
    raw_chirp = generate_chirp()
    shaped_chirp = apply_envelope(raw_chirp)
    print(f"Chirp generated: {len(shaped_chirp)} samples, {DURATION*1000:.1f} ms, "
          f"{F_START/1000:.1f}kHz → {F_END/1000:.1f}kHz")
    play_chirp(shaped_chirp)