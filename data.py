import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

import data_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def get_unlabeled_dataloaders(gex_features_df, seed, batch_size):
    """CCLE as source domain, Xena(TCGA)+PDXE as target domain"""
    set_seed(seed)
    pdxe_df = gex_features_df.loc[gex_features_df.index.str.startswith('X-')]
    xena_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA-')]
    
    train_pdxe_df, test_pdxe_df = train_test_split(pdxe_df, test_size=0.1)
    train_xena_df, test_xena_df = train_test_split(xena_df, test_size=len(test_pdxe_df) / len(pdxe_df))

    pdxe_dataset = TensorDataset(torch.from_numpy(pdxe_df.values.astype('float32')))
    xena_dataset = TensorDataset(torch.from_numpy(xena_df.values.astype('float32')))
    test_pdxe_dateset = TensorDataset(torch.from_numpy(test_pdxe_df.values.astype('float32')))
    test_xena_dateset = TensorDataset(torch.from_numpy(test_xena_df.values.astype('float32')))

    pdxe_dataloader = DataLoader(pdxe_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_pdxe_dataloader = DataLoader(test_pdxe_dateset, batch_size=batch_size, shuffle=True)
    xena_data_loader = DataLoader(xena_dataset, batch_size=batch_size, shuffle=True)
    test_xena_dataloader = DataLoader(test_xena_dateset, batch_size=batch_size, shuffle=True)

    return (pdxe_dataloader, test_pdxe_dataloader), (xena_data_loader, test_xena_dataloader)


def get_pdxe_labeled_dataloader_generator(gex_features_df, batch_size, drug, seed=2025,
                                         nan_flag=False, n_splits=5):

    drugs_to_keep = [item.lower() for item in drug]
    drugs_to_keep = ["gemcitabine-50mpk" if item == "gemcitabine" else item for item in drugs_to_keep]

    # Filter by beginning string "X"
    pdxe_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('X')]
    # read file
    pdxe_labeled_df = pd.read_table(data_config.pdxe_label_file, header=0, index_col=0)
    pdxe_labeled_df = pdxe_labeled_df.loc[:, drugs_to_keep]
    # Filter all NA values
    pdxe_labeled_df = pdxe_labeled_df.dropna(how='all')
    intersection_index = pdxe_gex_feature_df.index.intersection(pdxe_labeled_df.index)

    pdxe_labeled_df = pdxe_labeled_df.loc[intersection_index, drugs_to_keep]
    pdxe_labeled_gex_feature_df = pdxe_gex_feature_df.loc[intersection_index]
    masked_df = pdxe_labeled_df
    masked_df = (~np.array(masked_df.isnull())).astype(int)
    if nan_flag == False:
        pdxe_labeled_df = pdxe_labeled_df.replace(np.nan, -1)
    
    kfold = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_index, test_index in kfold.split(pdxe_labeled_gex_feature_df.values, pdxe_labeled_df.values, masked_df):
        train_labeled_pdxe_df, test_labeled_pdxe_df,  = pdxe_labeled_gex_feature_df.values[train_index], \
            pdxe_labeled_gex_feature_df.values[test_index]
        train_pdxe_labels, test_pdxe_labels = pdxe_labeled_df.values[train_index], pdxe_labeled_df.values[test_index]
        train_pdxe_mask, test_pdxe_mask = masked_df[train_index], masked_df[test_index]

        train_labeled_pdxe_dateset = TensorDataset(
            torch.from_numpy(train_labeled_pdxe_df.astype('float32')),
            torch.from_numpy(train_pdxe_labels.astype('float32')),
            torch.from_numpy(train_pdxe_mask.astype('float32'))
            )
        test_labeled_pdxe_df = TensorDataset(
            torch.from_numpy(test_labeled_pdxe_df.astype('float32')),
            torch.from_numpy(test_pdxe_labels.astype('float32')),
            torch.from_numpy(test_pdxe_mask.astype('float32'))
            )

        train_labeled_pdxe_dataloader = DataLoader(train_labeled_pdxe_dateset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_labeled_pdxe_dataloader = DataLoader(test_labeled_pdxe_df,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        yield train_labeled_pdxe_dataloader, test_labeled_pdxe_dataloader


def get_tcga_labeled_dataloaders(gex_features_df, drug, batch_size, nan_flag=False, tcga_cancer_type=None):
    if tcga_cancer_type is not None:
        raise NotImplementedError("Only support pan-cancer")

    drugs_to_keep = [item.lower() for item in drug]
    
    # Filter by beginning string "TCGA"
    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    # Take the first 12 characters of the original string as the new data id.
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    # Group by the new id and get the average of each column as the features.
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()

    tcga_labeled_df = pd.read_table(data_config.tcga_label_file, header=0, index_col=0)
    tcga_labeled_df = tcga_labeled_df.loc[:, drugs_to_keep]
    
    # Filter all NA values
    tcga_labeled_df = tcga_labeled_df.dropna(how='all')
    intersection_index = tcga_gex_feature_df.index.intersection(tcga_labeled_df.index)

    tcga_labeled_df = tcga_labeled_df.loc[intersection_index, drugs_to_keep]
    tcga_labeled_gex_feature_df = tcga_gex_feature_df.loc[intersection_index]
    masked_df = tcga_labeled_df
    masked_df = (~np.array(masked_df.isnull())).astype(int)
    if nan_flag == False:
        tcga_labeled_df = tcga_labeled_df.replace(np.nan, -1)

    labeled_tcga_dateset = TensorDataset(
        torch.from_numpy(tcga_labeled_gex_feature_df.values.astype('float32')),
        torch.from_numpy(tcga_labeled_df.values.astype('float32')),
        torch.from_numpy(masked_df.astype('float32'))
        )

    labeled_tcga_dataloader = DataLoader(labeled_tcga_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return labeled_tcga_dataloader


def get_lableled_dataloaders_generator(gex_features_df, drug, seed, batch_size, n_splits=5):
    set_seed(seed)
    pdxe_labeled_dataloader_generator = get_pdxe_labeled_dataloader_generator(gex_features_df=gex_features_df,
                                                                              batch_size=batch_size,
                                                                              drug=drug,
                                                                              seed=seed,
                                                                              n_splits=n_splits)

    test_labeled_dataloader = get_tcga_labeled_dataloaders(gex_features_df=gex_features_df,
                                                           batch_size=batch_size,
                                                           drug=drug)

    for train_labeled_pdxe_dataloader, test_labeled_pdxe_dataloader in pdxe_labeled_dataloader_generator:
        yield train_labeled_pdxe_dataloader, test_labeled_pdxe_dataloader, test_labeled_dataloader

    return pdxe_labeled_dataloader_generator


