import os
import torch
import numpy as np
from models import DomainDiscriminator, DrugClassifier, DomainAdversarialLoss
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve


def binary_focal_loss(inputs, targets, alpha=0.25, gamma=2, epsilon=1.e-9):
    multi_hot_key = targets
    logits = inputs
    logits = torch.sigmoid(logits)
    zero_hot_key = 1 - multi_hot_key
    loss = -alpha * multi_hot_key * torch.pow((1 - logits), gamma) * (logits + epsilon).log()
    loss += -(1 - alpha) * zero_hot_key * torch.pow(logits, gamma) * (1 - logits + epsilon).log()
    return loss.mean()


def domain_heuristic_loss(hiddens, focals):
    cosine_sim = F.cosine_similarity(hiddens, focals, dim=1)
    target = -torch.ones_like(cosine_sim)
    sim_loss = F.mse_loss(cosine_sim, target)
    focals = focals.reshape(-1)
    heuristic = torch.mean(torch.abs(focals))
    heuristic = heuristic + sim_loss
    return heuristic


def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)


def eval_epoch(epoch, model, loader, device):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_mask = [], [], []
    roc_list = []
    auprc_list = []
    f1_list = []
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        mask = (mask > 0)

        with torch.no_grad():
            yp = model(x)
            # loss_mat = nn.BCELoss()(yp, y.float())
            loss_mat = binary_focal_loss(yp, y.float())
            loss_mat = torch.where(
                mask, loss_mat,
                torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(mask)
            total_loss += loss
            y_true.append(y)
            y_pred.append(yp)
            y_mask.append(mask)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    y_mask = torch.cat(y_mask, dim=0).cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1.0) > 0 and np.sum(y_true[:, i] == 0.0) > 0:
            is_valid = (y_mask[:, i] > 0)
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
            auprc_list.append(auprc(y_true=y_true[is_valid, i], y_score=y_pred[is_valid, i]))
        else:
            print('{} is invalid'.format(i))

    return total_loss/len(loader), sum(roc_list)/len(roc_list), roc_list, auprc_list, y_true, y_pred, y_mask


def training(encoder, s_dataloader, v_dataloader, t_dataloader, drug, task_save_folder, params_str, **kwargs):
    base_network = DrugClassifier(backbone=encoder, num_classes=len(drug), bottleneck_dim=kwargs['latent_dim'], 
                                   finetune=True, sigmoid=False).to(kwargs['device'])
    domain_discri = DomainDiscriminator(in_feature=kwargs['latent_dim'], hidden_size=16).to(kwargs['device'])
    ad_net = DomainAdversarialLoss(domain_discri).to(kwargs['device'])
    
    optimizer = torch.optim.SGD(base_network.get_parameters() + domain_discri.get_parameters(), kwargs['lr'], momentum=0.9, weight_decay=kwargs["weight_decay"], nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: kwargs['lr'] * (1. + 0.001 * float(x)) ** (-0.75))

    best_auroc = -np.inf
    for epoch in range(800):
        base_network.train(True)
        ad_net.train(True)
        
        for step, s_batch in enumerate(s_dataloader):
            t_batch = next(iter(t_dataloader))
            s_x = s_batch[0].to(kwargs['device'])
            s_y = s_batch[1].to(kwargs['device'])

            t_x = t_batch[0].to(kwargs['device'])
            
            feat_s, out_s = base_network(s_x)
            feat_t, out_t = base_network(t_x)
            
            classifier_loss = binary_focal_loss(out_s, s_y.float())
            transfer_loss = ad_net(feat_s, feat_t)
            total_loss = classifier_loss + transfer_loss * kwargs['trade_off']

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()


        # validation and testing
        val_loss, val_avg_auc, val_auroc_list, val_auprc_list, val_y_true, val_y_pred, val_y_mask = eval_epoch(epoch=epoch,
                        model=base_network,
                        loader=v_dataloader,
                        device=kwargs['device'])
        test_loss, test_avg_auc, test_auroc_list, test_auprc_list, test_y_true, test_y_pred, test_y_mask = eval_epoch(epoch=epoch,
                        model=base_network,
                        loader=t_dataloader,
                        device=kwargs['device'])
        
        # early stopping
        if (val_avg_auc > best_auroc):
            print("Best model found at epoch {}: val_avg_auc = {:.4f} test_avg_auc = {:.4f}" .format(epoch, val_avg_auc, test_avg_auc))
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            best_auroc = val_avg_auc
            record_metric = [val_auroc_list, val_auprc_list, test_auroc_list, test_auprc_list]
            torch.save(base_network.state_dict(), os.path.join(kwargs['model_save_folder'], 'HDA.pt'))
            torch.save(ad_net.state_dict(), os.path.join(kwargs['model_save_folder'], 'AdversarialNetwork.pt'))
            np.save(os.path.join(kwargs['model_save_folder'], 'y_true.npy'), test_y_true)
            np.save(os.path.join(kwargs['model_save_folder'], 'y_pred.npy'), test_y_pred)
            np.save(os.path.join(kwargs['model_save_folder'], 'y_mask.npy'), test_y_mask)

    with open(os.path.join(task_save_folder, "{}_val_auroc.txt".format(params_str)), 'a') as opf:
        opf.write(",".join(list(map(str, record_metric[0]))))
        opf.write('\n')
    opf.close()
    with open(os.path.join(task_save_folder, "{}_val_auprc.txt".format(params_str)), 'a') as opf:
        opf.write(",".join(list(map(str, record_metric[1]))))
        opf.write('\n')
    opf.close()
    with open(os.path.join(task_save_folder, "{}_test_auroc.txt".format(params_str)), 'a') as opf:
        opf.write(",".join(list(map(str, record_metric[2]))))
        opf.write('\n')
    opf.close()
    with open(os.path.join(task_save_folder, "{}_test_auprc.txt".format(params_str)), 'a') as opf:
        opf.write(",".join(list(map(str, record_metric[3]))))
        opf.write('\n')
    opf.close()

    return base_network
