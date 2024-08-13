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
        #data that contains all paths
        self.data = []
        #iterate through every file
        for r,d,f in os.walk(self.path):
            #access all files
            all_files = [os.path.join(r,file) for file in f]
            self.data.extend(all_files)
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        #transform image and load image
        img = self.transforms(Image.open(self.data[idx]))
        return img
    

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import numpy as np
    import matplotlib.pyplot as plt
    data = CustomDataset(r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Images\images_64x64\cats",transforms.Compose([transforms.Resize(64),
                                     transforms.ToTensor(),
                                     transforms.CenterCrop(64),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    
    x = data[444]
    plt.imshow(np.transpose(np.clip(x,0,1),(1,2,0)))
    plt.show()