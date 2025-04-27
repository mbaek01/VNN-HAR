import os
import torch
import torch.nn as nn
from torch import optim
import yaml

from .baseline import Baseline, Baseline_Attn
from .vnn_mlp import VNN_MLP
from .sa_har import SA_HAR
from .deepconvlstm import DeepConvLSTM
from .deepconvlstm_attn import DeepConvLSTM_ATTN
from .vn_baseline_attn import VN_Baseline_Attn

class Model(object):
    def __init__(self, args):
        self.args = args

        self.device = self.acquire_device()

        self.optimizer_dict = {"Adam":optim.Adam}
        self.criterion_dict = {"MSE":nn.MSELoss, "CrossEntropy":nn.CrossEntropyLoss(reduction="mean")}

        self.model  = self.build_model().to(self.device)

    def acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Device: GPU, cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Device: CPU')
        return device
    
    def build_model(self):
        model = model_builder(self.args)
        return model.double()

    def select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError
        
        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def select_criterion(self):
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError
        
        criterion = self.criterion_dict[self.args.criterion]
        return criterion
    
    def forward(self, x):
        return self.model(x)
    

class model_builder(nn.Module):
    def __init__(self, args):
        super(model_builder, self).__init__()
        self.activation_fn_dict = {"relu":nn.ReLU, 
                                   "leakyrelu":nn.LeakyReLU, 
                                   #"vnleakyrelu":VLeakyReLU
                                   }
        config_file = open('./configs/model.yaml', mode='r')
        config = yaml.load(config_file, Loader=yaml.FullLoader)[args.model_name]

        # input shape for all models: (B, 1, L, C)

        if args.model_name == "baseline":
            self.model = Baseline(int(args.input_length * args.c_in),
                                args.num_classes,
                                self.activation_fn_dict[args.activation_fn]
                                )

            print("Model: baseline")
        
        elif args.model_name =="baseline_attn":
            self.model = Baseline_Attn(args.c_in,
                                        args.num_classes,
                                        config["nb_units"],
                                        self.activation_fn_dict[args.activation_fn]
                                )
            print("Model: baseline_attn")
        
        elif args.model_name == "vn_baseline_attn":
            self.model = VN_Baseline_Attn((args.batch_size, 1, args.input_length, args.c_in), # TODO: remove 1 if not needed
                                   args.num_classes,
                                   config["nb_units"],
                                   config["activation_fn"]
                                   )

        elif args.model_name == "vnn_mlp":
            self.model = VNN_MLP(args.batch_size, args.input_length, args.c_in, args.num_classes)

            print("Model: vnn_mlp")
        
        elif args.model_name == "sa_har":
            self.model = SA_HAR((1, args.input_length, args.c_in),
                                args.num_classes,
                                config)

            print("Model: sa_har")
        
        elif args.model_name == "deepconvlstm":
            self.model = DeepConvLSTM((1, args.input_length, args.c_in),
                                      args.num_classes,
                                      config)
        
        elif args.model_name == "deepconvlstm_attn":
            self.model = DeepConvLSTM_ATTN((1, args.input_length, args.c_in), 
                                            args.num_classes,
                                            # self.args.filter_scaling_factor,
                                            config)
            print("Model: deepconvlstm_attn")

        else:
            raise NotImplementedError

    def forward(self, x):
        y = self.model(x)
        return y
