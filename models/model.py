import os
import torch
import torch.nn as nn
from torch import optim
import yaml

from models.baseline import Baseline

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
            print('Using GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Using CPU')
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
        # self.args = args
        self.activation_fn_dict = {"relu":nn.ReLU, 
                                   "leakyrelu":nn.LeakyReLU, 
                                   #"vnleakyrelu":VLeakyReLU
                                   }

        # TODO: choosing model type if there are multiple
        config_file = open('./configs/model.yaml', mode='r')
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.model = Baseline(int(args.input_length * args.c_in),
                              args.num_classes,
                              self.activation_fn_dict[args.activation_fn]
                              )

        print("Using the Baseline model")

    def forward(self, x):
        y = self.model(x)
        return y