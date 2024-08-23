import torch
from torch import nn
from torchsummary import summary
import os
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import random
import numpy as np

# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

def init_weights(m):
    if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv2d):  # Prüfe auf Conv-Schichten
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):  # Für BatchNorm-Schichten
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show_audio_infos(path):
    all_files = []
    all_fs = []
    all_length = []
    for r,d,f in os.walk(path):
        files = [os.path.join(r,file) for file in f if file.endswith(".wav")]
        all_files.extend(files)
    #len files
    len_files = len(all_files)
    for file in all_files:
        infos = sf.info(file)
        all_fs.append(infos.samplerate)
        all_length.append(infos.duration)
    dataset_name = path.split("\\")[-1]
    print(
        f"""
        Some Statistiks about the {dataset_name} dataset given by the argument of this function:
        Amount of files found: {len_files} \n\tSamplerates found: {set(all_fs)} \n\tMean Duration: {sum(all_length)/len(all_length)} \n\tAll length: {sum(all_length)}sekunden
        """)

# show_audio_infos(r"C:\Users\analf\Desktop\Datasets_And_Results\Datasets\RS6")
# show_audio_infos(r"C:\Users\analf\Desktop\Datasets_And_Results\Datasets\MINST")
# show_audio_infos(r"F:\DataSets\Audio\Drums")

def plot_wave_spectrum(path):
    file = random.choice(os.listdir(path))
    file = os.path.join(path,file)
    data, fs = sf.read(file=file,always_2d=False)
    data_resampled = librosa.resample(y=data,orig_sr=fs,target_sr=16000)
    print(fs)
    fig,ax = plt.subplots(1,2)
    ax[0].plot(data_resampled)
    spec = librosa.stft(y=data_resampled,n_fft=256,hop_length=128)
    spec = librosa.amplitude_to_db(np.abs(spec),ref=np.max)
    librosa.display.specshow(data=spec,ax=ax[1],y_axis="linear",x_axis="time")
    plt.show()


def make_audio_suitable(path):
    #here the audio is splittet into 2 lists (beschleunigung and not)
    all_files = []
    for r,d,f in os.walk(path):
        all_files.extend([os.path.join(r,file) for file in f])
    beschleunigung = []

    for file in all_files:
        if file.lower().find("beschleunigung"):
            beschleunigung.append(file)
            all_files.remove(file)

    
    
    


if __name__ == "__main__":
    plot_wave_spectrum(r"C:\Users\analf\Desktop\Datasets_And_Results\Datasets\RS6")