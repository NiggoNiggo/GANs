import os
import librosa
import torchaudio.transforms as T
import torchaudio
import torch


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MnistAudio(Dataset):
    def __init__(self,
                 path:str,
                 num_classes,
                 transforms,
                 conditional:bool=False):
        super().__init__()
        self.path = path
        self.num_classes = num_classes
        self.transforms = transforms
        self.conditional = conditional
        print(self.conditional)
        if self.conditional == True:
            self.data = {x:[] for x in range(num_classes)}#contains label and data disct{str:list}
        self.load_data()
        
        
        
            
    def load_data(self):
        self.all_files = []
        for r,d,f in os.walk(self.path):
            self.all_files.extend([os.path.join(r,file) for file in f])
        if self.conditional == True: 
            print("Treu:" , self.conditional)   
            for file in self.all_files:
                #get first letter of the filename because it is the label
                label = int(file.split("\\")[-1][0])
                self.data[label].append(file)
                    
        
            
    
    def __len__(self):
        if self.conditional == True:
            return len(min([values for values in self.data.values()]))
        else:
            return len(self.all_files)
    
    def __getitem__(self,idx):
        if self.conditional == True:
            all_data = [values for values in self.data.values()]
            all_data = [item for sublist in all_data for item in sublist]
        else:
            all_data = self.all_files
        current_file = all_data[idx]
        current_label = int(current_file.split("\\")[-1][0])
        data, fs = torchaudio.load(current_file)

        data = self.transforms(data)
        if self.conditional == True:
            label = torch.tensor(current_label)
# mache mir hier einen plot des spektrogramms mit 1x128x128
            return data, label
        else:
            return data


    def plot_spectrogram(self, spectrogram):
        """ Utility function to plot the spectrogram. """
        spectrogram = spectrogram.squeeze().numpy()  # Convert to numpy array
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect='auto', origin='lower')
        plt.title("Spectrogram")
        plt.colorbar()
        plt.show()
    def get_num_classes(self):
        return self.num_classes

if __name__ == "__main__":
    dset = dataset = MnistAudio(path=r"F:\DataSets\Audio\MINST",
                                num_classes=10,)
    print(len(dset))
    loader = DataLoader(dataset,1,shuffle=True)
    x,y = next(iter(loader))
    print(x,y)