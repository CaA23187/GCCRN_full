import torch
from torch._C import device
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

import os
import json
import time

import numpy as np
import soundfile as sf
import librosa
import scipy.signal as sps
from scipy.special import expit
import matplotlib.pyplot as plt

json_path = './json_file/'
dataset_path = '/data/liuy/aec_dataset_100msDelay_2_2/'


win_size = 320
win_shift = 160
fft_num = win_size

# a_hp = [1 - 1.9999, 0.99600]
# b_hp = [1 - 2, 1]
K = 5
C = 0.1
p = 0
EPS = 1e-10


class ToTensor(object):
    def __call__(self, x, type):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return torch.IntTensor(x)


class MyDataset_new(Dataset):
    def __init__(self, json_dir, flag, compress_cIRM=False):
        assert flag in ['train_whole', 'valid_whole', 'test_whole']
        file_json_name = '_'.join(flag.split('/')) + '.json'
        with open(os.path.join(json_dir, file_json_name), 'r') as f:
            json_list = json.load(f)

        self.json_list = json_list[:]
        self.len = len(self.json_list)
        self.flag = flag.split('_')[0]
        self.compress_cIRM = compress_cIRM

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        wav_path = self.json_list[item % self.len]

        feats, labels = self.data_proce(wav_path)

        return feats, labels

    def data_proce(self, wav_path):
        to_tensor = ToTensor()

        farEnd_file_name = wav_path.split('/')[-1]
        name_split = farEnd_file_name.split('mic')

        farEnd_file_name = name_split[0] + 'lpb' + name_split[1]
        mic_file_name = name_split[0] + 'mic' + name_split[1]
        target_file_name = name_split[0] + 'clean' + name_split[1]

        file_path = '/'.join(wav_path.split(dataset_path)[-1].split('/')[:-2])

        # load Wavs
        farEnd_wav, fs = sf.read(os.path.join(dataset_path, file_path, 'lpb', farEnd_file_name))
        mic_wav, _ = sf.read(os.path.join(dataset_path, file_path, 'mic', mic_file_name))
        target_wav, _ = sf.read(os.path.join(dataset_path, file_path, 'clean', target_file_name))

        # # high pass filter
        # farEnd_wav = sps.lfilter(b_hp, a_hp, farEnd_wav)
        # mic_wav = sps.lfilter(b_hp, a_hp, mic_wav)
        # target_wav = sps.lfilter(b_hp, a_hp, target_wav)

        # STFT
        farEnd_stft = librosa.stft(farEnd_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T
        mic_stft = librosa.stft(mic_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T
        target_stft = librosa.stft(target_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T

        # convert np.array to torch.Tensor
        farEnd_complex = to_tensor(np.stack([farEnd_stft.real, farEnd_stft.imag], axis=1), 'float')
        mic_complex = to_tensor(np.stack([mic_stft.real, mic_stft.imag], axis=1), 'float')
        target_complex = to_tensor(np.stack([target_stft.real, target_stft.imag], axis=1), 'float')

        feat = torch.cat((farEnd_complex, mic_complex), dim=1)

        # cal compress cIRM
        if (self.compress_cIRM):
            # labels
            Mr = (mic_stft.real * target_stft.real
                  + mic_stft.imag * target_stft.imag) / (mic_stft.real ** 2
                                                         + mic_stft.imag ** 2 + EPS)
            Mi = (mic_stft.real * target_stft.imag
                  - mic_stft.imag * target_stft.real) / (mic_stft.real ** 2
                                                         + mic_stft.imag ** 2 + EPS)
            cIRMr = K * (2 * expit(C * Mr) - 1)
            cIRMi = K * (2 * expit(C * Mi) - 1)
            cIRM = np.stack([cIRMr, cIRMi], axis=1)

            label = to_tensor(cIRM, 'float')
        else:
            label = to_tensor(target_complex, 'float')

        return feat, label


if __name__ == '__main__':
    batch_size = 32
    num_workers = 4

    train_dataset = MyDataset_new(json_dir=json_path, flag='train_whole')
    valid_dataset = MyDataset_new(json_dir=json_path, flag='valid_whole')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4)

    t1 = time.time()
    n = 0
    for feats, labels in train_dataloader:
        print('feats: ', feats.size())
        print('feats: ', labels.size())
        break
