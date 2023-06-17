#Copyright (c) Meta Platforms, Inc. and affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.


import torch
import argparse
import librosa as librosa
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import os
import pandas as pd
import glob
from tqdm import tqdm
from model import NORESQA
from scipy import signal

# save_model_path = 'models/' # default
save_model_path = 'loaded_models/' # my config

CONFIG_PATH = save_model_path + 'wav2vec_small.pt'

def argument_parser():
    """
    Get an argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--metric_type', help='NORESQA->0, NORESQA-MOS->1', default=1, type=int)
    parser.add_argument('--GPU_id', help='GPU Id to use (-1 for cpu)', default=-1, type=int)
    parser.add_argument('--save_name', help='File name to save csv', default="result.csv", type=str)
    parser.add_argument('--mode', choices=['file', 'list'], help='predict noresqa for test file with another file (mode = file) as NMR or, with a database given as list of files (mode=list) as NMRs', default='file', type=str)
    parser.add_argument('--test_file', help='test speech file', required=False, type=str, default='sample_clips/noisy.wav')
    parser.add_argument('--nmr', help='for mode=file, path of nmr speech file. for mode=list, path of text file which contains list of nmr paths', required = False, type=str, default='sample_clips/clean.wav')
    return parser

args = argument_parser().parse_args()


# Noresqa model
model = NORESQA(output=40, output2=40, metric_type = args.metric_type, config_path = CONFIG_PATH)

# Loading checkpoint
if args.metric_type==0:
    model_checkpoint_path = save_model_path + 'model_noresqa.pth'
    state = torch.load(model_checkpoint_path,map_location="cpu")['state_base']
elif args.metric_type == 1:
    model_checkpoint_path = save_model_path + 'model_noresqa_mos.pth'
    state = torch.load(model_checkpoint_path,map_location="cpu")['state_dict']

pretrained_dict = {}
for k, v in state.items():
    if 'module' in k:
        pretrained_dict[k.replace('module.','')]=v
    else:
        pretrained_dict[k]=v
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict)

# change device as needed
# device
if args.GPU_id >=0 and torch.cuda.is_available():
    device = torch.device("cuda:{}".format(args.GPU_id))
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

sfmax = nn.Softmax(dim=1)
# function extraction stft
def extract_stft(audio, sampling_rate = 16000):

    fx, tx, stft_out = signal.stft(audio, sampling_rate, window='hann',nperseg=512,noverlap=256,nfft=512)
    stft_out = stft_out[:256,:]
    feat = np.concatenate((np.abs(stft_out).reshape([stft_out.shape[0],stft_out.shape[1],1]), np.angle(stft_out).reshape([stft_out.shape[0],stft_out.shape[1],1])), axis=2)
    return feat

# noresqa and noresqa-mos prediction calls
def model_prediction_noresqa(test_feat, nmr_feat):

    intervals_sdr = np.arange(0.5,40,1)

    with torch.no_grad():
        ranking_frame,sdr_frame,snr_frame = model(test_feat.permute(0,3,2,1),nmr_feat.permute(0,3,2,1))
        # preference task prediction
        ranking = sfmax(ranking_frame).mean(2).detach().cpu().numpy()
        pout = ranking[0][0]
        # quantification task
        sdr = intervals_sdr * (sfmax(sdr_frame).mean(2).detach().cpu().numpy())
        qout = sdr.sum()

    return pout, qout

def model_prediction_noresqa_mos(test_feat, nmr_feat):

    with torch.no_grad():
        score = model(nmr_feat,test_feat).detach().cpu().numpy()[0]

    return score

# reading audio clips
def audio_loading(path,sampling_rate=16000):

    audio, fs = librosa.load(path, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)

    if fs != sampling_rate:
        audio = librosa.resample(audio,fs,sampling_rate)

    return audio


# function checking if the size of the inputs are same. If not, then the reference audio's size is adjusted
def check_size(audio_ref,audio_test):

    if len(audio_ref) > len(audio_test):
        print('Durations dont match. Adjusting duration of reference.')
        audio_ref = audio_ref[:len(audio_test)]

    elif len(audio_ref) < len(audio_test):
        print('Durations dont match. Adjusting duration of reference.')
        while len(audio_test) > len(audio_ref):
            audio_ref = np.append(audio_ref, audio_ref)
        audio_ref = audio_ref[:len(audio_test)]

    return audio_ref, audio_test


# audio loading and feature extraction
def feats_loading(test_path, ref_path=None, noresqa_or_noresqaMOS = 0):

    if noresqa_or_noresqaMOS == 0 or noresqa_or_noresqaMOS == 1:

        audio_ref = audio_loading(ref_path)
        audio_test = audio_loading(test_path)
        audio_ref, audio_test = check_size(audio_ref,audio_test)

        if noresqa_or_noresqaMOS == 0:
            ref_feat = extract_stft(audio_ref)
            test_feat = extract_stft(audio_test)
            return ref_feat,test_feat
        else:
            return audio_ref, audio_test


if args.mode == 'file': # nmr이 1개인 경우
    if os.path.isdir(args.test_file): # test_file 개수가 여러개일 경우(폴더 경로 입력받음)
        test_file_list = glob.glob(args.test_file + '*')
        for test_file in test_file_list:
            nmr_feat,test_feat = feats_loading(test_file, args.nmr, noresqa_or_noresqaMOS = args.metric_type)
            test_feat = torch.from_numpy(test_feat).float().to(device).unsqueeze(0)
            nmr_feat = torch.from_numpy(nmr_feat).float().to(device).unsqueeze(0)

            if args.metric_type == 0:
                noresqa_pout, noresqa_qout = model_prediction_noresqa(test_feat, nmr_feat)
                print('Probaility of the test speech cleaner than the given NMR =', noresqa_pout)
                print('NORESQA score of the test speech with respect to the given NMR =', noresqa_qout)

            elif args.metric_type == 1:
                mos_score = model_prediction_noresqa_mos(test_feat, nmr_feat)
                print('MOS score of the test speech (assuming NMR is clean) =', str(5.0-mos_score))
    
    else: # 원래대로 작동 (1개의 test_file에 대해서만)
        nmr_feat,test_feat = feats_loading(args.test_file, args.nmr, noresqa_or_noresqaMOS = args.metric_type)
        test_feat = torch.from_numpy(test_feat).float().to(device).unsqueeze(0)
        nmr_feat = torch.from_numpy(nmr_feat).float().to(device).unsqueeze(0)

        if args.metric_type == 0:
            noresqa_pout, noresqa_qout = model_prediction_noresqa(test_feat, nmr_feat)
            print('Probaility of the test speech cleaner than the given NMR =', noresqa_pout)
            print('NORESQA score of the test speech with respect to the given NMR =', noresqa_qout)

        elif args.metric_type == 1:
            mos_score = model_prediction_noresqa_mos(test_feat, nmr_feat)
            print('MOS score of the test speech (assuming NMR is clean) =', str(5.0-mos_score))

elif args.mode == 'list':

    with open(args.nmr) as f:
        df = pd.DataFrame()
        p_column_name_list, q_column_name_list, s_column_name_list  = [], [], []
        all_pout_list, all_qout_list, all_mos_list = [], [], []
        for ln in tqdm(f):
            pout_list, qout_list, score_list = [], [], []
            file_name_list = [] # file_name_list 받는거 변경하기 temp = [] for xx in test_file_list: temp.append("/".join(xx.split('/')[-3:]))
            if os.path.isdir(args.test_file): # test_file 개수가 여러개일 경우(폴더 경로 입력받음)
                test_file_list = glob.glob(args.test_file + '*')
                # test_file_list = os.listdir(args.test_file)
                for test_file in tqdm(test_file_list):
                    print()
                    file_name_list.append("/".join(test_file.split('/')[-3:]))
                    nmr_feat,test_feat = feats_loading(test_file, ln.strip(), noresqa_or_noresqaMOS = args.metric_type)
                    test_feat = torch.from_numpy(test_feat).float().to(device).unsqueeze(0)
                    nmr_feat = torch.from_numpy(nmr_feat).float().to(device).unsqueeze(0)

                    if args.metric_type==0:
                        pout, qout = model_prediction_noresqa(test_feat,nmr_feat)
                        pout_list.append(pout)
                        qout_list.append(qout)
                        print("Test_file name: " + "/".join(test_file.split('/')[-3:]))
                        print(f"Prob. of test cleaner than {ln.strip()} = {pout}. Noresqa score = {qout}")

                    elif args.metric_type == 1:
                        score = model_prediction_noresqa_mos(test_feat,nmr_feat)
                        score_list.append(5-score)
                        print(f"MOS of test with respect to clean {ln.strip()} = {5-score}")
                    
            else: # 원래대로 작동 (1개의 test_file에 대해서만)
                nmr_feat,test_feat = feats_loading(args.test_file, ln.strip(), noresqa_or_noresqaMOS = args.metric_type)
                test_feat = torch.from_numpy(test_feat).float().to(device).unsqueeze(0)
                nmr_feat = torch.from_numpy(nmr_feat).float().to(device).unsqueeze(0)

                if args.metric_type==0:
                    pout, qout = model_prediction_noresqa(test_feat,nmr_feat)
                    pout_list.append(pout)
                    qout_list.append(qout)
                    print(f"Prob. of test cleaner than {ln.strip()} = {pout}. Noresqa score = {qout}")

                elif args.metric_type == 1:
                    score = model_prediction_noresqa_mos(test_feat,nmr_feat)
                    print(f"MOS of test with respect to clean {ln.strip()} = {5-score}")
    
            # 하나의 nmr이 끝날때마다 현재 작업 경로에 결과 csv 저장
            if len(pout_list) != 0: # metric_type:0(NORESQA)으로 실행한 경우
                p_column_name_list.append('p/' + "/".join(ln.strip().split('/')[-2:]))
                q_column_name_list.append('q/' + "/".join(ln.strip().split('/')[-2:]))
                all_pout_list.append(pout_list)
                all_qout_list.append(qout_list)
                # df['p/' + "/".join(ln.strip().split('/')[-2:])] = pout_list
                # df['q/' + "/".join(ln.strip().split('/')[-2:])] = qout_list
            else:
                s_column_name_list.append('M_s/' + "/".join(ln.strip().split('/')[-2:]))
                all_mos_list.append(score_list)
                # df['M_s/' + "/".join(ln.strip().split('/')[-2:])] = score_list
    
    df['test_file_name'] = file_name_list
    if args.metric_type == 0:
        for pidx, p in enumerate(p_column_name_list):
            df[p] = all_pout_list[pidx]
            # for pouts in all_pout_list:
            #     df[p] = pouts
        for qidx, q in enumerate(q_column_name_list):
            df[q] = all_qout_list[qidx]
            # for qouts in all_qout_list:
            #     df[q] = qouts
        df.to_csv(args.save_name, index=False) # index는 행에 표시되는 프레임index
    else: 
        for sidx, s in enumerate(s_column_name_list):
            df[s] = s_column_name_list[sidx]        
        df.to_csv(args.save_name, index=False)
    
    
        