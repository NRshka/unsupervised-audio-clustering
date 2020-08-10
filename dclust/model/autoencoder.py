import torch
from torch import nn


class TRAE(nn.Module):
    def __init__(
            self, num_layers: int,
            input_dim = 128,
            hidden_dim = 256,
            device='cpu'
        ):
        super().__init__()
        self.num_layers = num_layers
        self.device = device

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        self.embedding = nn.Linear(hidden_dim, input_dim*num_layers)
        self.decoder = nn.GRU(input_dim, input_dim, num_layers, dropout=0.2, batch_first=True, bidirectional=True)
        self.recon_layer = nn.Linear(2*input_dim, input_dim)

    def forward(self, x):
        '''Get input spectrogram (batch_size, freq, timestep) and return reconstructed tensor'''
        init_state = self.get_features(x)
        init_state = init_state.reshape(2*self.num_layers, x.size(0), -1)
        target = x.flip(1)

        # start decoding from 0
        decoder_inputs = target[:, :target.size(1) - 1]
        zeros = torch.zeros(target.size(0), 1, target.size(-1)).to(self.device)
        decoder_inputs = torch.cat([zeros, decoder_inputs], 1)

        recon, state = self.decoder(decoder_inputs, init_state)
        recon = torch.tanh(self.recon_layer(recon))

        return recon

    
    def get_features(self, x):
        enc_out, hidden = self.encoder(x)
        features = torch.tanh(self.embedding(hidden))
        return features.reshape(x.size(0), -1)
