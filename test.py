import numpy as np
import sounddevice as sd

sd.default.device = 'pipewire'

FS = 48000
DURATION = 1.5        # 1.5 full seconds — impossible to miss
FREQ = 440             # 440 Hz = musical note A, very audible

t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
tone = 0.5 * np.sin(2 * np.pi * FREQ * t)   # 0.5 = half volume, avoids distortion

print("Playing a loud 440Hz tone for 1.5 seconds...")
sd.play(tone, FS)
sd.wait()
print("Done.")