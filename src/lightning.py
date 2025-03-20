import gc
import os
from collections import defaultdict
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src._torch import CounterfactualRegressionTorch, DragonNetTorch
from src.directory import log_dir
from src.metrics import mmd, targeted_regularization_loss, dragonnet_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pl.seed_everything(40)


def get_trainer(model_name, checkpoint_callback, monitor='val_loss', mode='min', max_epochs=100, **kwargs):
    # set up logging
    logger = CSVLogger(save_dir=log_dir, name=model_name)

    trainer_kwargs = dict(
        precision="bf16-mixed",
        accelerator='auto',
        logger=logger,
        callbacks=[
            EarlyStopping(monitor=monitor, mode=mode, patience=5),
            checkpoint_callback
        ],
        log_every_n_steps=1,
        max_epochs=max_epochs,
    )

    trainer_kwargs.update(kwargs)

    # set up trainer
    trainer = pl.Trainer(
        **trainer_kwargs
    )

    return trainer


def get_checkpoint_callback(model_name, dir_path, monitor='val_loss', mode='min'):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=dir_path,
        filename=model_name + '-{epoch:002d}-{val_loss:.2f}',
        save_top_k=1)

    return checkpoint_callback


def get_log_dir_path(model_name):
    dir_path = os.path.join(log_dir, model_name)
    if not os.path.isdir(dir_path):
        version = '0'
    else:
        version = str(int(sorted(os.listdir(dir_path))[-1].replace('version_', '')) + 1)
    dir_path = os.path.join(dir_path, f'version_{version}')

    return dir_path

def get_logger(model_name, project_name='CPH_200B', wandb_entity='furtheradu', dir_path='..',**kwargs):
    if kwargs.get('disable_wandb'):
        print("wandb logging is disabled.")
        return None
    else:
        logger = pl.loggers.WandbLogger(
            project=project_name,
            entity=wandb_entity,
            group=model_name,
            dir=dir_path,
            **kwargs
        )
        return logger

class CounterfactualRegressionLightning(pl.LightningModule):
    def __init__(self, 
                 input_features, 
                 r_hidden_dim=200,
                 h_hidden_dim=100,
                 alpha=1.0,
                 learning_rate=1e-3,
                 complexity_lambda=1.0,
                 outcome_type='continuous'):
        super().__init__()
        valid_outcome_types = ['continuous', 'binary']
        assert outcome_type in valid_outcome_types, f'outcome_type must be in {valid_outcome_types}'
        self.save_hyperparameters()

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.input_features = input_features
        self.input_dim = len(self.input_features)
        self.r_hidden_dim = r_hidden_dim
        self.h_hidden_dim = h_hidden_dim
        self.outcome_type = outcome_type
        self.complexity_lambda = complexity_lambda
        
        if self.outcome_type == 'continuous':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            # self.loss_fn = log_loss
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        # structures to hold outputs/metrics
        self.outputs = defaultdict(list)
        self.metric_dict = defaultdict(dict)

        self.model = CounterfactualRegressionTorch(input_dim=self.input_dim,
                                                  r_hidden_dim=self.r_hidden_dim,
                                                  h_hidden_dim=self.h_hidden_dim)

        # init weights
        self.model.apply(self.init_weights)

    def forward(self, x, t):
        return self.model(x,t)

    def configure_optimizers(self):
        h_weight_decay = 1e-2
        hypothesis_params = []
        other_params = []

        for name, param in self.named_parameters():
            if 'h0' in name or 'h1' in name:
                hypothesis_params.append(param)
            else:
                other_params.append(param)

        # define optimizer with different weight decay for each group
        optimizer = optim.Adam([
            {'params': hypothesis_params, 'weight_decay': h_weight_decay},
            {'params': other_params, 'weight_decay': 0}
        ], lr=self.learning_rate)

        return optimizer
        
    def get_factual_loss(self, y_pred, Y, T, threshold=.5):
        u = self.trainer.datamodule.u
        w_i = (T / (2 * u) + (1 - T) / (2 * (1 - u)))
        # if self.outcome_type == 'binary':
        #     y_pred = (y_pred > threshold).float()
        loss = self.loss_fn(y_pred, Y)
        factual_loss = (w_i * loss).mean() # [~torch.any(loss.isnan(),dim=1)]
        return factual_loss

    @staticmethod
    def safely_to_numpy(tensor):
        return tensor.to(torch.float).cpu().numpy()
    
    @staticmethod
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    @staticmethod
    def get_model_input(batch):
        return batch['X'].squeeze(dim=1).double(), batch['Y'].double(), batch['T'].double()
    
    def step(self, batch, batch_idx, stage):
        X, Y, T = self.get_model_input(batch)

        out = self.model(X,T)
        phi_x, y0, y1 = out['phi_x'], out['y0'], out['y1']
        
        y_pred = y1 * T + y0 * (1 - T)
        phi_x_0 = phi_x[(T == 0).squeeze()]
        phi_x_1 = phi_x[(T == 1).squeeze()]

        factual_loss = self.get_factual_loss(y_pred,Y,T)
        IPM_loss = mmd(phi_x_0, phi_x_1)
        model_complexity_loss = torch.Tensor([0]) * self.complexity_lambda
        loss = self.alpha * IPM_loss + factual_loss + model_complexity_loss
        
        # store outputs
        self.outputs[stage].append({'X':X, 
                                    'T':T,
                                    'Y':Y,
                                    "phi_x":phi_x,
                                    'y0':y0,
                                    'y1':y1})

        # log metrics
        if stage not in ['predict', 'test']:
            log_kwargs = dict(prog_bar=True, sync_dist=True)
            self.log(f'{stage}_IPM_loss', IPM_loss, **log_kwargs)
            self.log(f'{stage}_factual_loss', factual_loss, **log_kwargs)
            self.log(f'{stage}_loss', loss, **log_kwargs)

        return loss
    
    def on_epoch_end(self, stage):
        # concat outputs
        outputs_vars = ['X', 'T', 'Y', 'phi_x', 'y0', 'y1']
        X, T, Y, phi_x, y0, y1 = [torch.cat([o[x] for o in self.outputs[stage]]).squeeze() 
                               for x in outputs_vars]

        # calculate performance metrics
        y_pred = y1 * T + y0 * (1 - T)
        factual_loss = self.get_factual_loss(y_pred,Y,T)
        tau = (y1 - y0).mean()
        
        metric_dict = {
            f'{stage}_{self.loss_fn._get_name()}': factual_loss,
            f'{stage}_tau': tau
        }

        if stage not in ['predict']:  # log metrics
            log_kwargs = dict(prog_bar=True, sync_dist=True)
            self.log_dict(metric_dict, **log_kwargs)

        self.metric_dict[stage].update(metric_dict)
        
        # send tensors to numpy
        X, T, phi_x = [self.safely_to_numpy(x) for x in [X, T, phi_x]]        
        
        # get t-SNE embeddings
        embeddings = {}
        for x_name, x in dict(zip(['X', 'phi_x'], [X, phi_x])).items():
            embeddings[x_name] = TSNE(random_state=40).fit_transform(x)
        
        if self.logger.experiment and stage not in ['train']:
            for k, embedding in embeddings.items():
                # get plot name
                key = f'tSNE {k}, CFR alpha {self.alpha}, epoch{self.current_epoch}'
                
                # make figure
                fig = plt.figure(figsize=(10, 8))
                ax = plt.scatter(embedding[:, 0], embedding[:, 1], c=T) # TODO: suppress printing
                plt.colorbar(ax, label='treatment')
                plt.title(key)
                                
                # log figure
                self.logger.log_image(key=key, images=[fig])
                
                # suppress displaying
                plt.close(fig)

        # clear outputs
        self.outputs[stage] = []

        # clean space
        gc.collect()
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "predict")

    # def on_train_epoch_end(self):
    #     self.on_epoch_end('train')

    # def on_validation_epoch_end(self):
    #     self.on_epoch_end('val')

    def on_test_epoch_end(self):
        self.on_epoch_end('test')

    # def on_predict_epoch_end(self):
    #     self.on_epoch_end('predict')
    
class DragonNetLightning(pl.LightningModule):
    def __init__(self, 
                 input_features, 
                 sr_hidden_dim=200,
                 co_hidden_dim=100,
                 alpha=1.0,
                 beta=1.0,
                 learning_rate=1e-5,
                 target_reg=False):
        super().__init__()
        self.save_hyperparameters()

        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.input_features = input_features
        self.input_dim = len(self.input_features)
        self.sr_hidden_dim = sr_hidden_dim
        self.co_hidden_dim = co_hidden_dim
        self.target_reg = target_reg
        
        # structures to hold outputs/metrics
        self.outputs = defaultdict(list)
        self.metric_dict = defaultdict(dict)

        self.model = DragonNetTorch(input_dim=self.input_dim,
                                    sr_hidden_dim=self.sr_hidden_dim,
                                    co_hidden_dim=self.co_hidden_dim)
        
        if self.target_reg:
            self.epsilon = nn.Parameter(torch.randn(1) / 10, requires_grad=True)
        
        # init weights
        self.model.apply(self.init_weights)

    def forward(self, x, t):
        return self.model(x,t)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=.9)
        return optimizer

    @staticmethod
    def safely_to_numpy(tensor):
        return tensor.to(torch.float).cpu().numpy()
    
    @staticmethod
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    @staticmethod
    def get_model_input(batch):
        return batch['X'].squeeze(dim=1).double(), batch['Y'].double(), batch['T'].double()

    def get_losses(self, y1, y0, t, Y, T):
        y1 = y1.squeeze()
        y0 = y0.squeeze()
        t = t.squeeze()
        Y = Y.squeeze()
        T = T.squeeze()
        
        dragon_net_loss = dragonnet_loss(y1, y0, t, Y, T, alpha=self.alpha)
        if self.target_reg:
            target_reg_loss = targeted_regularization_loss(y1, y0, t, Y, T, eps=self.epsilon, beta=self.beta)
        else:
            target_reg_loss = torch.Tensor([0])
        loss = dragon_net_loss + target_reg_loss
        return {'loss': loss, 'dragon_net_loss': dragon_net_loss, 'targeted_regularization_loss':target_reg_loss}
    
    def step(self, batch, batch_idx, stage):
        X, Y, T = self.get_model_input(batch)
        
        # get outputs
        out = self.model(X)
        t, y0, y1 = out['t'], out['y0'], out['y1']
        
        # get loss
        losses = self.get_losses(y1, y0, t, Y, T)
        loss = losses['loss']
        dragon_net_loss = losses['dragon_net_loss']
        target_reg_loss = losses['targeted_regularization_loss']
        
        # store outputs
        self.outputs[stage].append({'y0':y0,
                                    'y1':y1,
                                    't':t,
                                    'T':T,
                                    'Y':Y})

        # log metrics
        if stage not in ['predict', 'test']:
            log_kwargs = dict(prog_bar=True, sync_dist=True)
            self.log(f'{stage}_dragon_net_loss', dragon_net_loss, **log_kwargs)
            self.log(f'{stage}_target_regularization_loss', target_reg_loss, **log_kwargs)
            self.log(f'{stage}_loss', loss, **log_kwargs)

        return loss
    
    def on_epoch_end(self, stage):
        # concat outputs
        outputs_vars = ['Y', 'T', 'y1', 'y0', 't']
        Y, T, y1, y0, t = [torch.cat([o[x] for o in self.outputs[stage]]).squeeze() for x in outputs_vars]
        
        # get losses
        losses = self.get_losses(y1, y0, t, Y, T)
        loss = losses['loss']
        dragon_net_loss = losses['dragon_net_loss']
        target_reg_loss = losses['targeted_regularization_loss']
        
         # exclude data points with propensity score outside [.01,.99]
        included = (.01 <= t).squeeze() * (t <= .99).squeeze()
        y0 = y0[included]
        y1 = y1[included]
        T = T[included]

        # calculate performance metrics
        tau = (y1 - y0).mean()

        metric_dict = {
            f'{stage}_tau': tau,
            f'{stage}_loss': loss,
            f'{stage}_dragon_net_loss': dragon_net_loss,
            f'{stage}_targeted_regularization_loss': target_reg_loss
        }

        if stage not in ['predict']:  # log metrics
            log_kwargs = dict(prog_bar=True, sync_dist=True)
            self.log_dict(metric_dict, **log_kwargs)

        self.metric_dict[stage].update(metric_dict)

        # clear outputs
        self.outputs[stage] = []

        # clean space
        gc.collect()
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "predict")

    # def on_train_epoch_end(self):
    #     self.on_epoch_end('train')

    # def on_validation_epoch_end(self):
    #     self.on_epoch_end('val')

    def on_test_epoch_end(self):
        self.on_epoch_end('test')

    # def on_predict_epoch_end(self):
    #     self.on_epoch_end('predict')
