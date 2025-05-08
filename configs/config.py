import argparse
import os
import torch
import yaml

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Rotation-Invariant HAR Classification using Vector Neuron Network')
    parser.add_argument('-d', '--data_name', default='pamap2', type=str, help='Name of the Dataset')

    parser.add_argument('-m', '--model_name', default='baseline_attn', type=str, help="Name of the Model") 
    '''
    Available Models: deepconvlstm_attn,
                      deepconvlstm,
                      sa_har,
                      vn_sa_har
    '''

    parser.add_argument('-n', '--train', default=True,  type=str2bool, help="perform training")
    parser.add_argument('-t', '--test', default=True, type=str2bool, help='perform testing')
    parser.add_argument('-g', '--gpu', default=0, type=int, help="gpu index number")
    parser.add_argument('--train_rot', type=str, default='aligned', help='Rotation augmentation to input data in training [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--test_rot', type=str, default='aligned', help='Rotation augmentation to input data in testing [default: aligned]',
                    choices=['aligned', 'z', 'so3'])

    args = parser.parse_args()

    config_file = open('configs/data.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config[args.data_name]

    args.data_path = os.path.join("datasets", config['filename'])
    args.to_save_path = "saved"

    # if only args.test == True, need to specify the folder of the testing models
    args.test_path = ""

    args.use_gpu = True if torch.cuda.is_available() else False
    args.use_multi_gpu = True
    args.devices = "0,1,2,3,4,5,6,7" # available gpus

    args.optimizer = "Adam"
    args.criterion = "CrossEntropy"
    # args.exp_mode = "LOCV"
    args.datanorm_type = "standardization"
    args.activation_fn = "relu"

    # training settings
    args.train_epochs = 150
    args.learning_rate = 0.0001
    args.learning_rate_patience = 7
    args.learning_rate_factor = 0.1
    args.early_stop_patience = 20
    args.batch_size = 512 # 256
    args.shuffle = True
    args.drop_last = False
    args.train_vali_quote = 0.90
    args.f_in = 1 # input filter channel size

    window_seconds = config["window_seconds"]
    args.windowsize = int(window_seconds * config["sampling_freq"])
    args.input_length = args.windowsize
    args.c_in = config["num_channels"]
    args.sampling_freq = config["sampling_freq"]
    args.num_classes = config["num_classes"]
    args.filter_scaling_factor = 1
    args.sensor_select = ["acc"]
    args.seed = 10
    args.filtering = True
    args.freq1 = 0.001
    args.freq2 = 25.0

    ## saved pickle file paths - data preprocessing
    # 1) preprocessed data x and y for all subject
    # 2) window indices 
    args.pkl_save_path = os.path.join("datasets", args.data_name, f"window_size_{args.windowsize}")

    return args
