import argparse
import datetime
import os
import time
import zipfile
from tqdm import tqdm

import requests
import yaml

# This file uses code from:
# Saif Mahmud, M. T. H. Tonmoy, Kishor Kumar Bhaumik, A. M. Rahman, M. A. Amin, M. Shoyaib,
# Muhammad Asif Hossain Khan, and A. Ali.
# "Human Activity Recognition from Wearable Sensor Data Using Self-Attention."
# Proceedings of the 24th European Conference on Artificial Intelligence (ECAI 2020), 2020.
# Source: https://github.com/saif-mahmud/self-attention-HAR

# run 
# python dataset_download.py --dataset pamap2 --unzip

def get_dataset(url: str, file_name: str, unzip: bool): # data_directory: str, file_name: str, 
    # if not os.path.exists('datasets/'):
    #     os.mkdir('datasets/')

    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    data_directory = "datasets/"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    if not os.path.exists(os.path.join(data_directory, file_name)):
        print(f'GETTING DATASET [{file_name}] ...')

        response = requests.get(url, stream=True)
        data_file = open(os.path.join(data_directory, file_name), 'wb')

        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024

        for chunk in tqdm(iterable=response.iter_content(chunk_size=chunk_size), total=total_size / chunk_size,
                          unit='KB', unit_scale=True, unit_divisor=chunk_size):
            data_file.write(chunk)

        data_file.close()

        if unzip:
            print(f'Unzipping [{file_name}] ...')
            with zipfile.ZipFile(os.path.join(data_directory, file_name), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(data_directory, file_name.split('.')[0]))
                os.remove(os.path.join(data_directory, file_name))

        print('\n---DATASET DOWNLOAD COMPLETE---')

    else:
        print(f'Requested dataset exists in {data_directory}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self Attention Based HAR Model Training')
    parser.add_argument('-d', '--dataset', default='pamap2', type=str, help='Name of Dataset for Model Training')
    parser.add_argument('-z', '--unzip', action='store_true', help='Unzip downloaded dataset')
    args = parser.parse_args()

    config_file = open('configs/data.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    get_dataset(url=config[args.dataset]['source'],
                file_name=config[args.dataset]['destination'], unzip=args.unzip) #  data_directory=config['data_dir']['raw'],
