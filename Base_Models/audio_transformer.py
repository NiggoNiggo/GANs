import torchaudio.transforms as T
import torch.nn.functional as F
import torch

class SpecGANTransformer:
    def __init__(self, n_fft, win_length, hop_length, target_length, target_freq_bins,target_fs):
        self.stft = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2,normalized=True)
        self.to_amplitude = T.AmplitudeToDB()
        self.target_length = target_length
        self.target_freq_bins = target_freq_bins
        self.target_fs = target_fs

    
    def rescale(self,data):
        return data
    
    def clip(self,data):
        return data 


    def __call__(self, data,fs):
        data = data.float()
        if fs != self.target_fs:
            resample =  T.Resample(orig_freq=fs,new_freq=self.target_fs)
            data = resample(data)
        # Compute spectrogram
        spec = torch.abs(self.stft(data))
        spec = self.to_amplitude(spec)
        
        

        # Ensure the spectrogram has the correct number of frequency bins
        if spec.size(1) > self.target_freq_bins:
            # Truncate the frequency bins
            spec = spec[:, :self.target_freq_bins, :]
        elif spec.size(1) < self.target_freq_bins:
            # Pad the frequency bins
            padding = (0, 0, 0, self.target_freq_bins - spec.size(1))  # (left, right, top, bottom)
            spec = F.pad(spec, padding)
        
        # Ensure the spectrogram has the correct number of time steps
        if spec.size(2) > self.target_length:
            # Truncate the time steps
            spec = spec[:, :, :self.target_length]
        elif spec.size(2) < self.target_length:
            # Pad the time steps
            padding = (0, self.target_length - spec.size(2))  # (left, right)
            spec = F.pad(spec, padding)
        
        # #normalize spektrum
        # mean = torch.mean(spec, dim=-1,keepdim=True)
        # std = torch.std(spec,dim=-1,keepdim=True)
        
        # spec = (spec-mean)/std

        # spec = torch.clamp(spec,min=-3*std,max=3*std)

        # # spec = spec/(3*std)
    
        
        return spec
    



#transformer für audio preprocessing für WaveGAN
class WaveNormalizer:
    def __init__(self):
        pass

    def normalize_waveform(self,x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        normalized_x = (2*(x-x_min))/(x_max-x_min)-1
        return normalized_x

    def denormalize_waveform(self,x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        denormalized_x = ((x+1)*(x_max-x_min))/2+x_min
        return denormalized_x
    
    def make_same_length(self, x, length:int=16384):
        current_length = x.shape[-1]
        if current_length > length:
            # Schneide das Signal ab, wenn es zu lang ist
            x = x[..., :length]
        elif current_length < length:
            # Füge Zero-Padding hinzu, wenn es zu kurz ist
            padding = torch.zeros(x.shape[:-1] + (length - current_length,))
            x = torch.cat((x, padding), dim=-1)
        return x


    def __call__(self,x):
        x = self.normalize_waveform(x)
        x = self.make_same_length(x)
        return x