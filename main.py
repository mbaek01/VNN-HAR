import argparse
import os
import datetime
import torch
import yaml

from configs.config import get_args
from dataloaders.data_loader import PAMAP2, get_data
from models.model import Model
from utils import set_seed, get_setting_name
from train.trainer import Trainer, test_predictions

if __name__ == '__main__': 
    # args in configs/config.py
    args = get_args()

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.timestamp = timestamp

    # model_path = os.path.join(args.to_save_path, f"{args.model_name}_{timestamp}")

    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    
    # if not os.path.exists(args.test_save_path):
    #     os.makedirs(args.test_save_path)

    # Random Seed 
    set_seed(args.seed)
    print("Random Seed: ", args.seed)

    # Log file setting
    setting = get_setting_name(args) # includes timestamp
    curr_save_path = os.path.join(args.to_save_path, setting)
    
    if not os.path.exists(curr_save_path):
        os.makedirs(curr_save_path)

    # testing each subject as test subject 
    for test_sub in range(1,9):
        args.test_sub = test_sub

        # Dataset
        dataset = PAMAP2(args, test_sub)

        # TODO: cross-validation from here on 
        dataset.update_train_val_test_keys()

        train_loader = get_data(dataset, args.batch_size, flag = "train")
        valid_loader = get_data(dataset, args.batch_size, flag = "valid")
        test_loader = get_data(dataset, args.batch_size, flag = "test") # sub_id = test_sub 

        print(f"Dataset: {args.data_name} Loaded") 

        # Perform Training
        if args.train: 
            # Model Initialization
            model = Model(args)
            print("Using Model: ", args.model_name)

            # Trainer Initialization
            trainer = Trainer(args, model, curr_save_path)

            # Training starts
            print("Training the Model...")
            model = trainer.train(train_loader, valid_loader)

            print("Training completed!")

        if args.test: 
            print(f"Testing with Model - {args.model_load_name} of {args.model_name} | test subject: {test_sub}")

            acc, f_w, f_macro, f_micro = test_predictions(args, test_loader, curr_save_path, test_sub)
            

            