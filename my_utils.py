import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--option', help='make_clean_txt: 0, ', required=True, type=int)
args = parser.parse_args()

def make_clean_txt(data_path, save_path, save_txt): #
    data_list = os.listdir(data_path)
    data_list.sort()
    with open(save_path + save_txt, "a") as f:
        for data in data_list:
            save = data_path + '/' + data
            f.write(f'{save}\n')

if args.option == 0: # make_clean_txt
    clean_folder_path = 'clean_data/'
    folder_list = os.listdir(clean_folder_path)
    for folder in folder_list:
        make_clean_txt(clean_folder_path+folder, './', 'clean_data_list.txt')
            
