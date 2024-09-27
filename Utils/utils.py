from torch import nn
import os
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import random
import numpy as np
import pandas as pd


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

    
    
def make_conditional_dataset(path_a:str,path_b:str):
    class_1 = []
    class_2 = []
    #add filenames to class 1 list 
    for r,d,f in os.walk(path_a):
        class_1.extend([os.path.join(r,file) for file in f if file.endswith(".wav")])
    df_1 = pd.DataFrame({"Filename":class_1,"Label":[0 for _ in range(len(class_1))]})
    #add filenames to class 2 list
    for r,d,f in os.walk(path_b):
        class_2.extend([os.path.join(r,file) for file in f if file.endswith(".wav")])
    df_2 = pd.DataFrame({"Filename":class_2,"Label":[1 for _ in range(len(class_2))]})
    df = pd.concat([df_1,df_2],axis=0)
    print(df.head())
    df.to_csv("labeled_filenames.csv",index=False)


if __name__ == "__main__":

    make_conditional_dataset(r"H:\Datasets\RS6\Beschleunigung_noise",
                             r"H:\Datasets\RS6\snippets_clean")

