import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GCCRN_full import GCCRN
from data import ToTensor
import soundfile as sf

import numpy as np
import librosa

from data import win_size, win_shift, fft_num, K, C, EPS


class ToTensor(object):
    def __call__(self, x, type):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return torch.IntTensor(x)


def data_proce(net, lpb,mic,out):
    to_tensor = ToTensor()
    print('wav_path: ', lpb,mic,out)

    farEnd_wav, fs = sf.read(lpb)
    mic_wav, fs = sf.read(mic)

    farEnd_stft = librosa.stft(farEnd_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T
    mic_stft = librosa.stft(mic_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T

    Yr = np.real(mic_stft)
    Yi = np.imag(mic_stft)

    farEnd_complex = to_tensor(np.stack([farEnd_stft.real, farEnd_stft.imag], axis=1), 'float')
    mic_complex = to_tensor(np.stack([mic_stft.real, mic_stft.imag], axis=1), 'float')

    feat = torch.cat((farEnd_complex, mic_complex), dim=1)
    feat = feat.unsqueeze(0)

    output = net(feat)
    print('output size: ', output.size())

    M = output.squeeze()
    M = torch.clamp(M, -0.98 * K, 0.98 * K, out=None)
    M = M.numpy()

    Mr = M[:, 0, :]
    Mi = M[:, 1, :]

    if compress_cIRM:
        Mr = -1 / C * np.log(2 * K / (Mr + K) - 1 + EPS)
        Mi = -1 / C * np.log(2 * K / (Mi + K) - 1 + EPS)

        Sr = Mr * Yr - Mi * Yi
        Si = Mr * Yi + Mi * Yr
        Spec = Sr + 1j * Si
    else:
        Spec = Mr + 1j * Mi
    out_wav = librosa.istft(Spec.T, hop_length=win_shift, window='hanning')

    sf.write(out, out_wav, fs)


compress_cIRM = True
if __name__ == "__main__":

    lpb_file_list = ['lpb.wav']
    mic_file_list = ['mic.wav']
    out_file_list = ['out.wav']

    DEVICE = torch.device('cpu')
    net = GCCRN()

    # load parameters
    net.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('./GCCRN_full.pt', map_location=DEVICE).items()},
        strict=False)

    net.eval()  # prep model for evaluation
    p_list = []
    with torch.no_grad():
        for lpb, mic, out in zip(lpb_file_list, mic_file_list, out_file_list):
            data_proce(net, lpb,mic,out)


