import itertools
import torch
from torch import nn
    

class CFR_RepresentationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, n_layers=3):
        super().__init__()
        assert n_layers >= 2, 'n_layers must be >= 2'

        middle_layers = list(
            itertools.chain.from_iterable(
                [(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU()) for _ in range(n_layers-2)]
            )
        )
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            *middle_layers,
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        ]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class CFR_HypothesisNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, n_layers=3):
        super().__init__()
        assert n_layers >= 2, 'n_layers must be >= 2'

        middle_layers = list(
            itertools.chain.from_iterable(
                [(nn.Linear(hidden_dim, hidden_dim),nn.ELU()) for _ in range(n_layers-2)]
            )
        )
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            *middle_layers,
            nn.Linear(hidden_dim, 1),
        ]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class CounterfactualRegressionTorch(nn.Module):
    def __init__(self, input_dim, r_hidden_dim=200, h_hidden_dim=100):
        super().__init__()

        self.phi = CFR_RepresentationNetwork(input_dim, r_hidden_dim)
        self.h1 = CFR_HypothesisNetwork(r_hidden_dim +1, h_hidden_dim)
        self.h0 = CFR_HypothesisNetwork(r_hidden_dim + 1, h_hidden_dim)

        self.double()

    def forward(self, x, t):
        phi_x = self.phi(x)
        h_input = torch.cat([phi_x, t], dim=1) # concatenate representation with treatment
        y0 = self.h1(h_input)
        y1 = self.h0(h_input)
        return {'phi_x': phi_x, 'y0': y0, 'y1':y1}
    

class DragonNetTorch(nn.Module):
    def __init__(self, input_dim, sr_hidden_dim=200, co_hidden_dim=100, dropout_rate=0.2):
        super().__init__()

        self.shared_representation = CFR_RepresentationNetwork(input_dim, sr_hidden_dim)

        self.outcome_head_t1 = nn.Sequential(
            nn.Linear(sr_hidden_dim, co_hidden_dim // 2),
            nn.ELU(),
            nn.Linear(co_hidden_dim // 2, 1)
        )

        self.outcome_head_t0 = nn.Sequential(
            nn.Linear(sr_hidden_dim, co_hidden_dim // 2),
            nn.ELU(),
            nn.Linear(co_hidden_dim // 2, 1)
        )

        self.treatment_head = nn.Sequential(
            nn.Linear(sr_hidden_dim, 1),
            nn.Sigmoid()
        )
                
        self.double()

    def forward(self, x):
        shared_representation = self.shared_representation(x)
        y1_hat = self.outcome_head_t1(shared_representation)
        y0_hat = self.outcome_head_t0(shared_representation)
        t_hat = self.treatment_head(shared_representation)
        
        return {'y1':y1_hat, 'y0':y0_hat, 't':t_hat}