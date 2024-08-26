from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self,
                 path:str,
                 transforms):
        super().__init__()
        self.path = path #path to data
        self.transforms = transforms
        self.labels = []
        #data that contains all paths
        self.data = []
        #iterate through every file
        for r,d,f in os.walk(self.path):
            #access all files
            self.labels.append(d)
            all_files = [os.path.join(r,file) for file in f]
            self.data.extend(all_files)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        #transform image and load image
        img = self.transforms(Image.open(self.data[idx]))
        label = self.data[idx].split("\\")[-2]
        return img, label
    

