import os
import glob
import argparse
import soundfile as sf
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_lab_dir', required=True, type=str, help='input directory with VAD labels')
    parser.add_argument('--in_wav_dir', required=True, type=str, help='input directory for wavs')
    parser.add_argument('--sample_rate', required=False, type=int, default=16000, help='sampling rate')
    args = parser.parse_args()
    
    
    lab_file_paths = glob.glob(os.path.join(args.in_lab_dir, '*.lab'))
    for lab_file_path in lab_file_paths:
        # read lab file
        labs = np.atleast_2d((np.loadtxt(lab_file_path, usecols=(0, 1)) * args.sample_rate).astype(int))

        # read wav file
        fn = os.path.basename(lab_file_path).replace('.lab', '')
        signal, _ = sf.read(f'{os.path.join(args.in_wav_dir, fn)}.wav')
        for segnum in range(len(labs)):
            seg = signal[labs[segnum, 0]:labs[segnum, 1]]
