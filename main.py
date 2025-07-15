import pandas as pd
import torch
import json
import os
import argparse
import data
import data_config
import pretraining, domain_training
from copy import deepcopy


def get_drug_params(drug):
    with open(os.path.join('drug_params.json'), 'r') as f:
        drug_params = json.load(f)
    if drug in drug_params:
        return drug_params[drug]
    else:
        print(f"Warning: Drug '{drug}' not found in config, using default parameters.")
        return drug_params['default']


def wrap_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])
    return aux_dict


def make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)


def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])


def main(args, drug, params_dict):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # load mixed gene expressions for both tcga patients and nibr pdxs
    gex_features_df = pd.read_pickle(data_config.gex_feature_file)

    # load traning params
    with open(os.path.join('default_params.json'), 'r') as f:
        training_params = json.load(f)
    training_params['unlabeled'].update(params_dict)
    training_params['labeled'].update(params_dict)
    param_str = dict_to_str(params_dict)
    method_save_folder = os.path.join('model_save')

    training_params.update({'device': device,
                            'input_dim': gex_features_df.shape[-1],
                            'model_save_folder': os.path.join(method_save_folder, drug[0], param_str),
                            'retrain_flag': args.retrain_flag,
                            'norm_flag': args.norm_flag})

    task_save_folder = os.path.join(f'{method_save_folder}', drug[0])
    make_dir(training_params['model_save_folder'])
    make_dir(task_save_folder)

    with open(os.path.join(task_save_folder, "{}_val_auroc.txt".format(param_str)), 'a') as opf:
        opf.write(",".join(drug))
        opf.write('\n')
    opf.close()
    with open(os.path.join(task_save_folder, "{}_val_auprc.txt".format(param_str)), 'a') as opf:
        opf.write(",".join(drug))
        opf.write('\n')
    opf.close()
    with open(os.path.join(task_save_folder, "{}_test_auroc.txt".format(param_str)), 'a') as opf:
        opf.write(",".join(drug))
        opf.write('\n')
    opf.close()
    with open(os.path.join(task_save_folder, "{}_test_auprc.txt".format(param_str)), 'a') as opf:
        opf.write(",".join(drug))
        opf.write('\n')
    opf.close()

    pdxe_dataloaders, tcga_dataloaders = data.get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2025,
        batch_size=training_params['unlabeled']['batch_size'])
    
    # start unlabeled training, obtain target shared encoder
    encoder, s_dsnae = pretraining.training(s_dataloaders=pdxe_dataloaders,
                                   t_dataloaders=tcga_dataloaders,
                                   **wrap_params(training_params,
                                                 type='unlabeled'))

    labeled_dataloader = data.get_lableled_dataloaders_generator(gex_features_df=gex_features_df,
                                                                seed=2025,
                                                                batch_size=training_params['labeled']['batch_size'],
                                                                drug=drug,
                                                                n_splits=args.n)

    i = 0
    for train_labeled_pdxe, test_labeled_pdxe, labeled_tcga in labeled_dataloader:
        i = i + 1
        print("Fold: ", i)
        ft_encoder = deepcopy(encoder) 
        
        print('--------------------', drug)
        print('PDXE training samples:', train_labeled_pdxe.dataset.tensors[1].shape)
        print('PDXE testing samples:', test_labeled_pdxe.dataset.tensors[1].shape)
        print('TCGA testing samples:', labeled_tcga.dataset.tensors[1].shape)

        domain_training.training(encoder=ft_encoder,
                                s_dataloader=train_labeled_pdxe,
                                v_dataloader=test_labeled_pdxe,
                                t_dataloader=labeled_tcga,
                                drug=drug,
                                task_save_folder=task_save_folder,
                                params_str=param_str,
                                **wrap_params(training_params, type='labeled'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining and Fine_tuning')
    parser.add_argument('--drug', dest='drug', nargs='?', default='Paclitaxel', choices=['Cetuximab', 'Paclitaxel', 'Gemcitabine'])
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()

    params_list=get_drug_params(args.drug)
    main(args=args, params_dict=params_list,
            drug = [args.drug]
        )
           
