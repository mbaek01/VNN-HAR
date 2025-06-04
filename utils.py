import datetime
import numpy as np
import pandas as pd
import random
import torch
from scipy.fftpack import fft,fftfreq,ifft
import scipy as sp
import yaml


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        
    def fit(self, df):
        if self.norm_type == "standardization":
            self.mean = df.mean(0)
            self.std = df.std(0)
        elif self.norm_type == "minmax":
            self.max_val = df.max()
            self.min_val = df.min()
        elif self.norm_type == "per_sample_std":
            self.max_val = None
            self.min_val = None
        elif self.norm_type == "per_sample_minmax":
            self.max_val = None
            self.min_val = None
        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')
        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

def components_selection_one_signal(t_signal,freq1,freq2,sampling_freq):
    """
    DC_component: f_signal values having freq between [-0.3 hz to 0 hz] and from [0 hz to 0.3hz] 
                                                                (-0.3 and 0.3 are included)
    
    noise components: f_signal values having freq between [-25 hz to 20 hz[ and from ] 20 hz to 25 hz] 
                                                                  (-25 and 25 hz inculded 20hz and -20hz not included)
    
    selecting body_component: f_signal values having freq between [-20 hz to -0.3 hz] and from [0.3 hz to 20 hz] 
                                                                  (-0.3 and 0.3 not included , -20hz and 20 hz included)
    """

    t_signal=np.array(t_signal)
    t_signal_length=len(t_signal) # number of points in a t_signal
    
    # the t_signal in frequency domain after applying fft
    f_signal=fft(t_signal) # 1D numpy array contains complex values (in C)
    
    # generate frequencies associated to f_signal complex values
    freqs=np.array(sp.fftpack.fftfreq(t_signal_length, d=1/float(sampling_freq))) # frequency values between [-25hz:+25hz]
    

    
    
    f_DC_signal=[] # DC_component in freq domain
    f_body_signal=[] # body component in freq domain numpy.append(a, a[0])
    f_noise_signal=[] # noise in freq domain
    
    for i in range(len(freqs)):# iterate over all available frequencies
        
        # selecting the frequency value
        freq=freqs[i]
        
        # selecting the f_signal value associated to freq
        value= f_signal[i]
        
        # Selecting DC_component values 
        if abs(freq)>freq1:# testing if freq is outside DC_component frequency ranges
            f_DC_signal.append(float(0)) # add 0 to  the  list if it was the case (the value should not be added)                                       
        else: # if freq is inside DC_component frequency ranges 
            f_DC_signal.append(value) # add f_signal value to f_DC_signal list
    
        # Selecting noise component values 
        if (abs(freq)<=freq2):# testing if freq is outside noise frequency ranges 
            f_noise_signal.append(float(0)) # # add 0 to  f_noise_signal list if it was the case 
        else:# if freq is inside noise frequency ranges 
            f_noise_signal.append(value) # add f_signal value to f_noise_signal

        # Selecting body_component values 
        if (abs(freq)<=freq1 or abs(freq)>freq2):# testing if freq is outside Body_component frequency ranges
            f_body_signal.append(float(0))# add 0 to  f_body_signal list
        else:# if freq is inside Body_component frequency ranges
            f_body_signal.append(value) # add f_signal value to f_body_signal list
    
    ################### Inverse the transformation of signals in freq domain ########################
    # applying the inverse fft(ifft) to signals in freq domain and put them in float format
    t_DC_component= ifft(np.array(f_DC_signal)).real
    t_body_component= ifft(np.array(f_body_signal)).real
    #t_noise=ifft(np.array(f_noise_signal)).real
    
    #total_component=t_signal-t_noise # extracting the total component(filtered from noise) 
    #                                 #  by substracting noise from t_signal (the original signal).
    

    #return (total_component,t_DC_component,t_body_component,t_noise) 
    return (t_DC_component,t_body_component) 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

class EarlyStopping:
    """Early stops the training if f1 macro, or validation loss, doesn't improve after a given patience."""
    def __init__(self, metric="f1_macro", patience=7, verbose=False, delta=0):
        """
        Args:
            metric (str): metric used for early stopping - f1_macro or valid_loss
                            Default: "f1_macro"
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
  
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.f1_macro_max = -np.inf
        self.delta = delta
        self.metric = str(metric).lower()

    def __call__(self, val_loss, model, path, f_macro, f_weighted = None, log=None):

        if self.metric == "valid_loss":
            score = -val_loss
        elif self.metric == "f1_macro":
            score = f_macro

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, f_macro, f_weighted)

        elif score < self.best_score + self.delta:
            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("new best score!!!!")
            if log is not None:
                log.write("new best score!!!! Saving model ... \n")
                log.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n")
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, f_macro, f_weighted)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, f_macro, f_weighted = None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.metric == "valid_loss":
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                self.val_loss_min = val_loss
            elif self.metric == "f1_macro":
                print(f'F1 Macro increased ({self.f1_macro_max:.6f} --> {f_macro:.6f}).  Saving model ...')
                self.f1_macro_max = f_macro

        torch.save(model.state_dict(), path+'/'+f'best_vali.pth')


class adjust_learning_rate_class:
    def __init__(self, args, verbose):
        self.patience = args.learning_rate_patience
        self.factor   = args.learning_rate_factor
        self.learning_rate = args.learning_rate
        self.args = args
        self.verbose = verbose
        self.val_loss_min = np.inf
        self.counter = 0
        self.best_score = None
    def __call__(self, optimizer, val_loss):
        # val_loss is a positive value, and smaller the better
        # bigger the score, the better
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.counter += 1
        elif score <= self.best_score :
            self.counter += 1
            if self.verbose:
                print(f'Learning rate adjusting counter: {self.counter} out of {self.patience}')
        else:
            if self.verbose:
                print("new best score!!!!")
            self.best_score = score
            self.counter = 0
            
        if self.counter == self.patience:
            self.learning_rate = self.learning_rate * self.factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                if self.verbose:
                    print('Updating learning rate to {}'.format(self.learning_rate))
            self.counter = 0

def get_setting_name(args):
    config_file = open('configs/model.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)[args.model_name]

    if args.model_name == "sa_har":
        setting = "sa_har_data_{}_seed_{}_window_size_{}_num_units_{}_lr_{}_rot_{}_lr_scheduler_{}_{}".format(
            args.data_name,
            args.seed,
            args.windowsize,
            config["nb_units"],
            args.learning_rate,
            "_".join([args.train_rot, args.test_rot]),
            args.learning_rate_adapter,
            args.timestamp
            )

    elif args.model_name == "vn_sa_har":
        setting = "vn_sa_har_data_{}_seed_{}_window_size_{}_num_units_{}_batch_size_{}_{}".format(
            args.data_name,
            args.seed,
            args.windowsize,
            config["nb_units"],
            args.batch_size,
            args.timestamp
            )

    elif args.model_name == "deepconvlstm":
        setting = "deepconvlstm_data_{}_seed_{}_windowsize_{}_cvfilter_{}_lstmfilter_{}_{}".format(
            args.data_name,
            args.seed,
            args.windowsize,
            config["nb_filters"],
            config["nb_units_lstm"],
            args.timestamp
            )
    
    elif args.model_name == "deepconvlstm_attn":
        setting = "deepconvlstm_attn_data_{}_seed_{}_windowsize_{}_cvfilter_{}_lstmfilter_{}_{}".format(
            args.data_name,
            args.seed,
            args.windowsize,
            config["nb_filters"],
            config["nb_units_lstm"],
            args.timestamp
            )

    # elif args.model_name == "eq_deepconvlstm":
    #     setting = "eq_deepconvlstm_attn_data_{}_seed_{}_windowsize_{}_nb_fields_{}_lstmfilter_{}_{}".format(
    #         args.data_name,
    #         args.seed,
    #         args.windowsize,
    #         config["nb_fields"],
    #         config["nb_units_lstm"],
    #         args.timestamp
    #     )

    elif args.model_name== "baseline_attn":
        setting = "baseline_attn_data_{}_nb_unit_{}_lr_{}_rot_{}_lr_scheduler_{}_seed_{}_{}".format(
            args.data_name,
            config["nb_units"],
            args.learning_rate,
            "_".join([args.train_rot, args.test_rot]),
            args.learning_rate_adapter,
            args.seed,
            args.timestamp
            )

    elif args.model_name== "vn_baseline_attn":
        setting = "vn_baseline_attn_data_{}_nb_unit_{}_lr_{}_rot_{}_lr_scheduler_{}_seed_{}_{}".format(
            args.data_name,
            config["nb_units"],
            args.learning_rate,
            "_".join([args.train_rot, args.test_rot]),
            args.learning_rate_adapter,
            args.seed,
            args.timestamp
            )
        
    elif args.model_name== "vn_inv_baseline_attn":
        setting = "vn_inv_baseline_attn_data_{}_nb_unit_{}_lr_{}_rot_{}_lr_scheduler_{}_seed_{}_{}".format(
            args.data_name,
            config["nb_units"],
            args.learning_rate,
            "_".join([args.train_rot, args.test_rot]),
            args.learning_rate_adapter,
            args.seed,
            args.timestamp
            )

    # elif args.model_name== "baseline":
    #     setting = "baseline_data_{}_seed_{}_windowsize_{}_{}".format(
    #         args.data_name,
    #         args.seed,
    #         args.windowsize,
    #         args.timestamp
    #         )

    # elif args.model_name== "vnn_mlp":
    #     setting = "vnn_mlp_data_{}_seed_{}_windowsize_{}_{}".format(
    #         args.data_name,
    #         args.seed,
    #         args.windowsize
    #         )

    

    else:
        raise NotImplementedError
    
    return setting
    
def vn_c_reshape(x, time_length):
    # For PAMAP only!!

    # Example input: (batch, 1, time_length, 9)
    # Original order: [x_hand, y_hand, z_hand, x_chest, y_chest, z_chest, x_ankle, y_ankle, z_ankle]
    channel_indices = [
        0, 3, 6,  # x for hand, chest, ankle
        1, 4, 7,  # y for hand, chest, ankle
        2, 5, 8   # z for hand, chest, ankle
    ]
    batch = x.size(0)

    # x is your input tensor of shape (batch, 1, time_length, 9)
    x_reordered = x[:, :, :, channel_indices]  # [batch, 1, time_length, 9]

    # Now reshape
    x_reshaped = x_reordered.reshape(batch, 1, time_length, 3, -1) # [batch, 1, time_length, 3(xyz), Channels // 3]

    return x_reshaped