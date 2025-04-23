import os
import datetime
import numpy as np

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

    # for test performance aggregation (if args.test==True)
    acc_list = []
    f_w_list = []
    f_macro_list = []
    f_micro_list = []

    # Dataset
    dataset = PAMAP2(args)
    print(f"Dataset: {args.data_name} Loaded") 

    # testing each subject as test subject 
    for test_sub in range(1,9):
        # updates the test subject for cross validation
        dataset.update_train_val_test_keys()
        print(f"Using subject {dataset.index_of_cv} as a test subject")

        # curr_save_path = model_name/(test_subject_num)
        curr_save_path = os.path.join(curr_save_path, str(dataset.index_of_cv))
        if not os.path.exists(curr_save_path):
            os.makedirs(curr_save_path)

        train_loader = get_data(dataset, args.batch_size, flag = "train")
        valid_loader = get_data(dataset, args.batch_size, flag = "valid")
        test_loader = get_data(dataset, args.batch_size, flag = "test") 

        # Perform Training
        if args.train: 
            # Model Initialization
            model = Model(args)

            # Trainer Initialization
            trainer = Trainer(args, model, curr_save_path)

            # Training starts
            print("Training the Model...")
            model = trainer.train(train_loader, valid_loader)

            print("Training completed!")

        # TODO: separate training and testing process into different for loops of subjects
        if args.test: 
            print(f"Testing with Model - {args.model_load_name} of {args.model_name} | test subject: {dataset.index_of_cv}")

            # Log setting for the test performance
            score_log_file_path = os.path.join(curr_save_path, "score.txt")            
            score_log = open(os.path.join(curr_save_path, "score.txt"), "a")

            print("Score Log File: ", score_log_file_path)

            acc, f_w, f_macro, f_micro = test_predictions(args, test_loader, curr_save_path, score_log)

            acc_list.append(acc)
            f_w_list.append(f_w)
            f_macro_list.append(f_macro)
            f_micro_list.append(f_micro)

            if test_sub == 8: 
                score_log.write(f"Model: curr_save_path \n"
                                f"Accuracy: mean={np.mean(acc_list):.7f}, std={np.std(acc_list):.7f}\n"
                                f"F1 Weighted: mean={np.mean(f_w_list):.7f}, std={np.std(f_w_list):.7f}\n"
                                f"F1 Macro: mean={np.mean(f_macro_list):.7f}, std={np.std(f_macro_list):.7f}\n"
                                f"F1 Micro: mean={np.mean(f_micro_list):.7f}, std={np.std(f_micro_list):.7f}\n")
