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
    def __init__(self, input_dim, sr_hidden_dim=200, co_hidden_dim=100, n_treatment_groups=2, discrete_outcome=True):
        super().__init__()
        
        self.n_treatment_groups = n_treatment_groups

        self.shared_representation = CFR_RepresentationNetwork(input_dim, sr_hidden_dim)

        outcome_final_layer = [nn.Sigmoid()] if discrete_outcome else []
        self.outcome_heads = nn.ModuleDict({
            str(k): nn.Sequential(
                    nn.Linear(sr_hidden_dim, co_hidden_dim // 2),
                    nn.ELU(),
                    nn.Linear(co_hidden_dim // 2, 1),
                    *outcome_final_layer
                ) 
             for k in range(self.n_treatment_groups)})

        # set treatment head params in binary or multi-treatment case
        if self.n_treatment_groups == 2:
            treatment_head_out = 1 
            activation = [nn.Sigmoid()]
        else:
            treatment_head_out = self.n_treatment_groups
            activation = [nn.Softmax(dim=1)] # alternatively, [] # unnormalized logits
            
        self.treatment_head = nn.Sequential(
            nn.Linear(sr_hidden_dim, treatment_head_out),
            *activation
        )
        
        self.epsilon = nn.Linear(1,1)

        self.double()

    def forward(self, x):
        shared_representation = self.shared_representation(x)
        ys_hat = {k: self.outcome_heads[str(k)](shared_representation) for k in range(self.n_treatment_groups)}
        t_hat = self.treatment_head(shared_representation)
        eps = self.epsilon(torch.ones_like(t_hat)[:, 0:1])
        return {'ys':ys_hat, 't':t_hat, 'eps':eps, 'phi_x': shared_representation}