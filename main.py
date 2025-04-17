import argparse
import os
import torch
import yaml

from dataloaders.data_loader import PAMAP2, get_data
from models.model import Model
from utils import set_seed, get_setting_name
from train.trainer import Trainer

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Rotation-Invariant HAR Classification using Vector Neuron Network')
    parser.add_argument('-d', '--data_name', default='pamap2', type=str, help='Name of the Dataset')
    parser.add_argument('-m', '--model_name', default='baseline_attn', type=str, help="Name of the Model") 
    # baseline, baseline_att, sa_har, deepconvlstm_attn

    config_file = open('configs/data.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    args = parser.parse_args()

    config = config[args.data_name]

    args.data_path = os.path.join("datasets",config['filename'])
    args.to_save_path = "saved"

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.use_multi_gpu = False

    # args.model_name = "baseline_attn" #baseline2
    args.optimizer = "Adam"
    args.criterion = "CrossEntropy"
    args.exp_mode = "LOCV"
    args.datanorm_type = "standardization" # None ,"standardization", "minmax", 
    args.activation_fn = "relu"

    # training setting
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
    # args.f_in = 1 # with additional filtering
    args.input_length = args.windowsize
    args.c_in = config["num_channels"]
    args.sampling_freq = config["sampling_freq"]
    args.num_classes = config["num_classes"]
    args.filter_scaling_factor = 1
    args.sensor_select = ["acc"] #gyro
    # args.pos_select = None

    args.seed = 10

    args.filtering = True

    args.freq1 = 0.3
    args.freq2 = 25.0

    # Random Seed 
    set_seed(args.seed)
    print("Random Seed: ", args.seed)

    # Log file setting
    setting = get_setting_name(args)
    path = os.path.join(args.to_save_path,'logs/'+setting)
    
    if not os.path.exists(path):
        os.makedirs(path)

    epoch_log_file_name = os.path.join(path, "epoch_log.txt") # cv_path for cross-validation
    score_log_file_name = os.path.join(path, "score.txt")

    epoch_log = open(epoch_log_file_name, "a")
    score_log = open(score_log_file_name, "a")

    print("Epoch Log File: ", epoch_log_file_name)
    print("Score Log File: ", score_log_file_name)

    # Dataset
    dataset = PAMAP2(args)

    # TODO: cross-validation from here on 
    dataset.update_train_val_test_keys()

    train_loader = get_data(dataset, args.batch_size, flag = "train")
    valid_loader = get_data(dataset, args.batch_size, flag = "valid")
    test_loader = get_data(dataset, args.batch_size, flag = "test")

    print(f"Dataset: {args.data_name} Loaded") 
    # train_steps = len(train_loader)

    # Model 
    model = Model(args)
    model.build_model()

    print("Using Model: ", args.model_name)

    # Initialize trainer
    trainer = Trainer(args, model, epoch_log)

    # Train the model
    print("Training the Model...")
    model = trainer.train(train_loader, valid_loader)

    print("Training completed!")
