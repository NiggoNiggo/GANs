import torchaudio.transforms as T
import torchvision.transforms as F
import torch

class SpecGANTransformer:
    def __init__(self, n_fft, win_length, hop_length, target_length, target_freq_bins,target_fs):
        self.stft = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None)
        self.target_length = target_length
        self.target_freq_bins = target_freq_bins
        self.target_fs = target_fs

    def __call__(self, data):
        data = data.float()
        # Compute spectrogram
        spec = torch.abs(self.stft(data))
        
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
        
        return spec
    