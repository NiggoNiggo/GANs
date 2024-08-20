import librosa
import torchaudio

import numpy as np
import os
import random
import matplotlib.pyplot as plt

root = r"C:\Users\analf\Desktop\Datasets_And_Results\Datasets\MINST"

file = random.choice(os.listdir(root))
path = os.path.join(root,file)
print(f"The file {file} was choiced")
data, fs = librosa.load(path,sr=16000)



spectrum = librosa.stft(y=data,n_fft=256,hop_length=128)
magnitude = librosa.amplitude_to_db(np.abs(spectrum),ref=np.max)

fix, ax = plt.subplots()

librosa.display.specshow(data=magnitude,ax=ax,y_axis="linear",sr=fs,hop_length=128,x_axis="time")
# fig.colorbar(img, ax=ax, format="%+2.f dB")

plt.show()