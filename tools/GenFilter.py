'''
generate the mel filters for NiuTrans.Speech.
python3 GenFilter.py -n_mels $num_mels -o $output_path
* `n_mels` - Num of mel filters.  v3 is 128 and others are 80. Default: 128.
* `o` - Path to save the mel filters.
'''

import librosa
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Tool to generate mel filter',
)
parser.add_argument('-n_mels', required=False, type=int, default="128", choices=[128, 80],
    help='the size of filter.')
parser.add_argument('-o', required=False, type=str,
    help='Path of filter output')
args = parser.parse_args()

#large-v3 is 128 and others are 80
N_MELS = args.n_mels 
mel_filter = librosa.filters.mel(sr=16000, n_fft=400, n_mels=N_MELS)
a = np.asarray(mel_filter)
np.savetxt(args.o, a, delimiter=",")
