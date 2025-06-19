import torch.nn as nn

from .attn import AttentionWithContext, EncoderLayer, AttentionWithContext2

class Baseline(nn.Module):
    def __init__(self, in_dim, out_dim, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn()

        self.fc1 = nn.Linear(in_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.ln1 = nn.LayerNorm(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.ln2 = nn.LayerNorm(512)


        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.ln3 = nn.LayerNorm(256)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.ln4 = nn.LayerNorm(128)

        self.fc5 = nn.Linear(128, out_dim)

    def forward(self, x):
        '''
        :param x: torch.Tensor
                Shape(batch, length, channel)
        '''
        x = x.view(x.size(0), -1)
        x = self.activation_fn(self.bn1(self.fc1(x)))
        x = self.activation_fn(self.bn2(self.fc2(x)))
        x = self.activation_fn(self.bn3(self.fc3(x)))
        x = self.activation_fn(self.bn4(self.fc4(x)))
        out = self.fc5(x)  
        return out

class Baseline_Attn(nn.Module):
    def __init__(self, input_shape, nb_classes, nb_units, activation_fn, attn_act_fn = "tanh"):
        super().__init__()
        self.time_length = input_shape[2]
        self.channel = input_shape[3]
        self.nb_units = nb_units
        self.activation_fn = activation_fn()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.channel, self.nb_units)
        # self.ln1 = nn.LayerNorm(self.nb_units)

        # self.fc2 = nn.Linear(self.nb_units, in_dim)
        # self.ln2 = nn.LayerNorm(in_dim)

        self.EncoderLayer1 = EncoderLayer( d_model = self.nb_units, n_heads =4 , d_ff = self.nb_units*4)
        self.EncoderLayer2 = EncoderLayer( d_model = self.nb_units, n_heads =4 , d_ff = self.nb_units*4)

        # self.AttentionWithContext = AttentionWithContext(self.nb_units, attn_act_fn)
        self.AttentionWithContext = AttentionWithContext2(self.nb_units, self.time_length)

        self.fc3 = nn.Linear(self.nb_units, 4*nb_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(4*nb_classes, nb_classes)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        # Shape: (B, 1, L, C); C = nb_units
        # x = self.activation_fn(self.fc2(x)) 

        x = x.squeeze(1)
        # Shape: (B, L, C)
        x = self.EncoderLayer1(x)
        x = self.EncoderLayer2(x)
        # Shape: (B, 1, L, C); C = nb_units

        x = self.AttentionWithContext(x)
        # Shape: (B, C); C = nb_units

        x = self.dropout(self.relu(self.fc3(x)))
        # Shape: (B, C) ; C = 4*nb_classes

        out = self.fc_out(x)
        # Shape: (B, C); C = nb_classes
        return out