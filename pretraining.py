import os
import torch
import numpy as np
from itertools import chain
from models import DSNAE, MLP

def eval_dsnae_epoch(model, data_loader, device):
    model.eval()
    avg_loss = 0
    for x_batch in data_loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            loss = model.loss_function(*(model(x_batch)))
            avg_loss += loss['loss'].cpu().detach().item() / x_batch.size(0)
    return avg_loss

def dsn_ae_train_step(s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, scheduler=None):
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    optimizer.zero_grad()
    loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.cpu().detach().item() / s_x.size(0)

def training(s_dataloaders, t_dataloaders, **kwargs):
    s_train_dataloader = s_dataloaders[0]
    s_test_dataloader = s_dataloaders[1]
    t_train_dataloader = t_dataloaders[0]
    t_test_dataloader = t_dataloaders[1]

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         drop=kwargs['drop']).to(kwargs['device'])
    shared_decoder = MLP(input_dim=2 * kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         drop=kwargs['drop']).to(kwargs['device'])
    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['drop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])
    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['drop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])


    thres = np.inf
    if kwargs['retrain_flag']:
        ae_params = [t_dsnae.private_encoder.parameters(),
                     s_dsnae.private_encoder.parameters(),
                     shared_decoder.parameters(),
                     shared_encoder.parameters()
                     ]
        ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
        # start dsnae pre-training
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            train_loss_all = 0
            val_loss_all = 0
            if epoch % 50 == 0:
                print(f'AE training epoch {epoch}')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                train_loss = dsn_ae_train_step(s_dsnae=s_dsnae,
                                                        t_dsnae=t_dsnae,
                                                        s_batch=s_batch,
                                                        t_batch=t_batch,
                                                        device=kwargs['device'],
                                                        optimizer=ae_optimizer)
                train_loss_all += train_loss
            s_val_loss = eval_dsnae_epoch(model=s_dsnae,
                                                 data_loader=s_test_dataloader,
                                                 device=kwargs['device'])
            t_val_loss = eval_dsnae_epoch(model=t_dsnae,
                                                 data_loader=t_test_dataloader,
                                                 device=kwargs['device'])
            val_loss_all = s_val_loss + t_val_loss
            if (val_loss_all < thres):
                thres = val_loss_all
                torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 's_dsnae.pt'))
                torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 't_dsnae.pt'))
    else:
        try:
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 't_dsnae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")
    t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 't_dsnae.pt')))  
    return t_dsnae.shared_encoder, s_dsnae