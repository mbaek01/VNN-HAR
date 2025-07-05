import os
from dotmap import DotMap
import numpy as np
import random
import yaml
import torch
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn

from dataloaders.data_loader import PAMAP2, get_data
from models.vn_dgcnn import VNDGCNNEncoder
from models.vn_attn import VNEncoderLayer
from models.vn_layers import VNLinear, VNLeakyReLU
from train.rotation import Rotate, apply_rotation
from utils import vn_c_reshape, vn_c_unreshape

def get_error(output_rotated, output_from_rot):
    delta = output_rotated - output_from_rot
    numerator = torch.norm(delta)
    denom = torch.norm(output_rotated)
    rel_error = numerator / denom

    error_mean = torch.mean(delta)

    print(f"Relative Error: {rel_error}")
    # print(f"Mean: {error_mean}")
    
    return rel_error.cpu().detach().numpy()

def random_rotation_matrix():
    theta = torch.rand(3) * 2 * torch.pi
    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta[0]), -torch.sin(theta[0])],
                        [0, torch.sin(theta[0]), torch.cos(theta[0])]], device=device)
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])],
                        [0, 1, 0],
                        [-torch.sin(theta[1]), 0, torch.cos(theta[1])]], device=device)
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0],
                        [torch.sin(theta[2]), torch.cos(theta[2]), 0],
                        [0, 0, 1]], device=device)
                        
    rot_mat = R_z @ R_y @ R_x
    return rot_mat.double()

class VN_T_Encoder(nn.Module):
    def __init__(self, nb_units):
        super().__init__()

        self.nb_units = nb_units

        self.fc1 = VNLinear(1, self.nb_units)
        self.vn_act = VNLeakyReLU(self.nb_units, share_nonlinearity=False, negative_slope=0.0)

        # Self-Attention Encoder
        self.VNEncoderLayer1 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)
        self.VNEncoderLayer2 = VNEncoderLayer(d_model = self.nb_units, n_heads=4 , d_ff = self.nb_units*4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.vn_act(x)

        x = self.VNEncoderLayer1(x)
        x = self.VNEncoderLayer2(x)

        return x

if __name__ == '__main__': 
    config_file = open('configs/data.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    args = DotMap()
    args.data_name = "pamap2"
    config = config[args.data_name]

    args.datapath = os.path.join("datasets", config['filename'])

    window_seconds = config["window_seconds"]
    args.windowsize = int(window_seconds * config["sampling_freq"])

    args.seed = 10
    args.batch_size= 32

    # Gravity Filtering
    args.filtering = True
    # Freq Range
    args.freq1 = 0.001
    args.freq2 = 25.0

    # Position and Sensor Type Filtering
    args.pos_select = ["hand"] # None
    args.sensor_select = ["acc"]    

    # Normalization Type
    args.datanorm_type = "standardization"

    # Preprocessed Data Save Path
    args.pkl_save_path = os.path.join("datasets", args.data_name, f"window_size_{args.windowsize}")

    # Train Valid Split Ratio
    args.train_vali_quote = 0.90


    # VN_DGCNN
    args.k = 20 # num_neighbors


    dataset = PAMAP2(args)

    k = 5
    input_dim = 1
    output_dim = 341
    device = "cuda:1"

    dg_encoder = VNDGCNNEncoder(num_neighbors=k, 
                            dims = [input_dim, 21, 21, 42, 85, output_dim]).to(device).double()
    
    t_encoder = VN_T_Encoder(nb_units=120).to(device).double()
    
    rel_error_list = []

    for test_sub in range(1,9):
        # updates the test subject for cross validation
        dataset.update_train_val_test_keys()
        print(f"Test Subject: {dataset.index_of_cv}")

        train_loader = get_data(dataset, args.batch_size, flag = "train")
        # valid_loader = get_data(dataset, args.batch_size, flag = "valid")
        # test_loader = get_data(dataset, args.batch_size, flag = "test") 

        rel_error_sub = []
        
        for batch_x1, _ in train_loader:
            batch_x1 = batch_x1.double().to(device)
            # batch_y = batch_y.long().to(device)

            # Rotation R
            trot = Rotate(batch_x1.shape[0], 'so3', device, torch.float64)

            # f(Rx)
            batch_x1 = vn_c_reshape(batch_x1, batch_x1.shape[2])         # (B, C, L, 3, D)

            B, C, L, _, D = batch_x1.shape
            batch_x1_reshaped = batch_x1.reshape(B, C*D, 3, L)           # (B, C*D, 3, L)

            batch_x1_rot = trot.transform_points(batch_x1_reshaped)

            # Test unreshape
            # batch_x1_rot = batch_x1_rot.reshape(B, C, L, 3, D)
            # batch_x1_rot = vn_c_unreshape(batch_x1_rot)       
            # batch_x1_rot = vn_c_reshape(batch_x1_rot, L)
            # batch_x1_rot = batch_x1_rot.reshape(B, C*D, 3, L)
            
            output_from_rot = dg_encoder(batch_x1_rot)

            # f(x)R
            output = dg_encoder(batch_x1_reshaped) 

            output_rotated = trot.transform_points(output) 
            
            rel_error = get_error(output_rotated, output_from_rot)
            rel_error_sub.append(rel_error)

            del batch_x1, output, output_from_rot, output_rotated
            torch.cuda.empty_cache()

        rel_error_list.append(np.mean(rel_error_sub))

    print(rel_error_list)


             