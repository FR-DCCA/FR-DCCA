import torch.nn as nn
import torch.nn.init as init


class Encoder(nn.Module):
    def __init__(self, layer_sizes, input_size: int, output_size: int):
        super(Encoder, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes + [output_size]
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id]),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1]),
                ))
        self.layers = nn.ModuleList(layers)

    def init_encoder(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                if i == len(self.layers)-1 and j == 1:
                    init.kaiming_normal_(self.layers[i][j].weight)
                elif i != len(self.layers)-1 and j == 0:
                    init.kaiming_normal_(self.layers[i][j].weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, layer_sizes, input_size: int, output_size: int):
        super(Decoder, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes + [output_size]
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id]),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1]),
                ))
        self.layers = nn.ModuleList(layers)

    def init_decoder(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                if i == len(self.layers)-1 and j == 1:
                    init.kaiming_normal_(self.layers[i][j].weight)
                elif i != len(self.layers)-1 and j == 0:
                    init.kaiming_normal_(self.layers[i][j].weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x





