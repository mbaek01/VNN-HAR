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

    # Available Models: baseline_attn, deepconvlstm_attn,
    parser.add_argument('-m', '--model_name', default='baseline_attn', type=str, help="Name of the Model") 
    
    parser.add_argument('-n', '--train', default=True,  type=str2bool, help="perform training")
    parser.add_argument('-t', '--test', default=False, type=str2bool, help='perform testing')

    args = parser.parse_args()

    config_file = open('configs/data.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config[args.data_name]

    args.data_path = os.path.join("datasets", config['filename'])
    args.to_save_path = "saved"

    # # test predictions args
    # args.model_load_name = "best_vali_2025-04-18_05-27-33"
    # args.model_load_path = os.path.join(args.to_save_path, f"{args.model_load_name}.pth")
    # args.test_save_path = os.path.join(args.to_save_path, args.model_name)

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 6
    args.use_multi_gpu = False

    args.optimizer = "Adam"
    args.criterion = "CrossEntropy"
    args.exp_mode = "LOCV"
    args.datanorm_type = "standardization"
    args.activation_fn = "relu"

    # training settings
    args.train_epochs = 150
    args.learning_rate = 0.0001
    args.learning_rate_patience = 7
    args.learning_rate_factor = 0.1
    args.early_stop_patience = 15
    args.batch_size = 256
    args.shuffle = True
    args.drop_last = False
    args.train_vali_quote = 0.90

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

    return args
