# https://www.youtube.com/watch?v=88FFnqt5MNI&ab_channel=ValerioVelardo-TheSoundofAI
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class EmotionSpeechDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device 
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples 

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label


    def _cut_if_necessary(self, signal):
        # signal -> tensor -> (1, num_samples) -> (1, 50000) -> (1, 16000)
        if signal.shape[1] > self.num_samples: 
            signal = signal[:, :self.num_samples]
        return signal 

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: # (2, 1000)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal 


    def _get_audio_sample_path(self, index):
        #fold = f"fold{self.annotations.iloc[index, 1]}"
        path = self.annotations.iloc[index, 2]
        return path 

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]


if __name__ == "__main__":
    ANNOTATIONS_FILE = '/Users/stephen/Emotion_Dectection/data/RAVDESS/metadata.csv'
    AUDIO_DIR = '/Users/stephen/Emotion_Dectection/data/RAVDESS/Audio_Speech_Actors_01-24/'
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000

    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate =SAMPLE_RATE, 
        n_fft = 1024, 
        hop_length = 512, 
        n_mels=64
    )


    emo = EmotionSpeechDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device )
    print(f"There are {len(emo)} samples in the dataset.")
    signal, label = emo[1]

