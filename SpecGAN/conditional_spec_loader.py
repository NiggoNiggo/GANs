import os
import librosa
import torchaudio.transforms as T
import torchaudio


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MnistAudio(Dataset):
    def __init__(self,
                 path:str,
                 num_classes,
                 transforms):
        super().__init__()
        self.path = path
        self.num_classes = num_classes
        self.transforms = transforms
        self.data = {x:[] for x in range(num_classes)}#contains label and data disct{str:list}
        print(self.data)
        self.load_data()
        
        
            
    def load_data(self):
        all_files = []
        for r,d,f in os.walk(self.path):
            all_files.extend([os.path.join(r,file) for file in f])    
        for file in all_files:
            #get first letter of the filename because it is the label
            label = int(file.split("\\")[-1][0])
            self.data[label].append(file)
            
        
            
    
    def __len__(self):
        return len(min([values for values in self.data.values()]))
    
    def __getitem__(self,idx):
        all_data = [values for values in self.data.values()]
        all_data = [item for sublist in all_data for item in sublist]
        print(all_data)
        current_file = all_data[idx]
        current_label = int(current_file.split("\\")[-1][0])
        data, fs = torchaudio.load(current_file)
        return self.transforms(data)


    def get_num_classes(self):
        return self.num_classes

if __name__ == "__main__":
    dset = dataset = MnistAudio(path=r"F:\DataSets\Audio\MINST",
                                num_classes=10,)
    print(len(dset))
    loader = DataLoader(dataset,1,shuffle=True)
    x,y = next(iter(loader))
    print(x,y)