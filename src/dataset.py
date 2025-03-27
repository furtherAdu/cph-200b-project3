import tqdm
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FromPandasDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_dict = {k: torch.tensor([v]) for k, v in self.dataset[idx].items()}
        return sample_dict


class XYTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=128,
            num_workers=11,
            treatment_col='T',
            outcome_col='Y',
            input_features=[],
            dataset_name='ihdp',
            features_to_standardize=[],
            raw_data=None,
            **kwargs):
        super().__init__()

        self.raw_data = raw_data
        self.n_samples = len(self.raw_data)
        self.dataset_name = dataset_name
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.u = None
        self.treatment_classes = None
        self.features_to_standardize = features_to_standardize
        self.scaler = StandardScaler()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.input_features = input_features
        self.feature_config = [*self.input_features]
        self.splits = dict(zip(['train', 'val', 'test'],
                               [[], [], []]))

        train_val_idx, test_idx = train_test_split(list(self.raw_data.index), random_state=12, test_size=.2)
        train_idx, val_idx = train_test_split(train_val_idx, random_state=12, test_size=.2)
        self.split_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    def setup(self, stage=None):
        self.u = self.raw_data[self.treatment_col].mean()
        self.treatment_classes = self.raw_data[self.treatment_col].unique().tolist()
        
        # standardize features
        if self.features_to_standardize:
            self.scaler.fit(self.raw_data.loc[self.split_idx['train'], self.features_to_standardize])
            self.raw_data[self.features_to_standardize] = self.scaler.transform(self.raw_data[self.features_to_standardize])

        # setup assumes data is one-hot encoded
        for idx, row in tqdm.tqdm(self.raw_data.iterrows(), desc=f'Processing {self.dataset_name} Dataset'):
            split = [k for k, v in self.split_idx.items() if idx in v][0]

            # add input features
            input_features = {'X': [row[k] for k in self.input_features]} if self.input_features else {}

            # add treatment and outcome info
            T_features = {'T': row[self.treatment_col]}
            Y_features = {'Y': row[self.outcome_col]}

            # create sample
            sample = {**T_features, **Y_features, **input_features}

            # add it to the split
            self.splits[split].append(sample)
        
        dataset_kwargs = {}

        if stage == 'fit':
            self.train = FromPandasDataset(self.splits['train'], **dataset_kwargs)
            self.val = FromPandasDataset(self.splits['val'], **dataset_kwargs)
        if stage == 'validate':
            self.val = FromPandasDataset(self.splits['val'], **dataset_kwargs)
        if stage == 'test':
            self.test = FromPandasDataset(self.splits['test'], **dataset_kwargs)
        if stage == 'predict':
            self.predict = FromPandasDataset(self.splits['test'], **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val.__len__(), num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test.__len__(), num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.predict.__len__(), num_workers=1)