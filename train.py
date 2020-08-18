import os
import argparse
import json
import torch
import torch.nn.functional as F
from models.cae import CAE
from models.cae_danet import CAE_DANet
from data import LibriMix, LIBRISPEECH_SPKID
from torch.utils.data import DataLoader
from losses.sisdr import PermInvariantSISDR
from losses.spk_loss import CircleLoss, convert_label_to_similarity
from tqdm import tqdm
import yaml
from pprint import pprint
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
import numpy as np
import math


EPS = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')
parser.add_argument('--stage', type=int, default=2,
                    help='stage1: cae pre-training, stage2: embedding net training')
parser.add_argument('--model_dir', default='sanet')
parser.add_argument('--cuda', type=str, nargs="+", default=['0'])
parser.add_argument('-train', '--train_metadata', type=str, nargs="+", default=['/storageNVME/fei/data/speech/Librimix/Libri2Mix/wav8k/min/metadata/mixture_train-100_mix_clean.csv'])
parser.add_argument('-tn', '--train_n_src', type=int, nargs="+", default=[2])
parser.add_argument('-val', '--val_metadata', type=str, nargs="+", default=['/storageNVME/fei/data/speech/Librimix/Libri2Mix/wav8k/min/metadata/mixture_dev_mix_clean.csv'])
parser.add_argument('-vn', '--val_n_src', type=int, nargs="+", default=[2])

def config_cae_path(cae_args):
    path = '{}_N{}_L{}_S{}_bias{}_{}'.format(cae_args['mask'],
                                             cae_args['n_filters'],
                                             cae_args['kernel_size'],
                                             cae_args['stride'],
                                             cae_args['bias'],
                                             cae_args['enc_act'])
    return path

def main(conf):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(conf['main_args']['cuda'])
    model_dir = conf['main_args']['model_dir']
    exp_dir = conf['main_args']['exp_dir']
    # Define Dataloader
    assert len(conf['main_args']['train_metadata']) == len(conf['main_args']['train_n_src'])
    train_gens = []
    for i in range(len(conf['main_args']['train_metadata'])):
        train_set = LibriMix(csv_path=conf['main_args']['train_metadata'][i],
                             sample_rate=conf['data']['sample_rate'],
                             n_src=conf['main_args']['train_n_src'][i],
                             segment=conf['data']['segment'])
        train_gen = DataLoader(train_set, shuffle=True,
                               batch_size=int(conf['training']['batch_size']*2/conf['main_args']['train_n_src'][i]),
                               num_workers=conf['training']['num_workers'],
                               drop_last=True)
        train_gens.append(train_gen)

    assert len(conf['main_args']['val_metadata']) == len(conf['main_args']['val_n_src'])
    val_gens = []
    for i in range(len(conf['main_args']['val_metadata'])):
        val_set = LibriMix(csv_path=conf['main_args']['val_metadata'][i],
                           sample_rate=conf['data']['sample_rate'],
                           n_src=conf['main_args']['val_n_src'][i],
                           segment=conf['data']['segment'])
        val_gen = DataLoader(val_set, shuffle=True,
                             batch_size=conf['training']['batch_size'],
                             num_workers=conf['training']['num_workers'],
                             drop_last=True)
        print(val_gen)
        val_gens.append(val_gen)
    SPKID = LIBRISPEECH_SPKID[conf['data']['subset']]

    # Loss functions
    loss_fn = dict()
    loss_fn['sisdr'] = PermInvariantSISDR(return_individual_results=True)
    loss_fn['spk_circle'] = CircleLoss(m=0.25, gamma=15)

    # Define model, optimizer + scheduler
    if conf['main_args']['stage'] == 1:
        model = CAE(conf['cae'])
        model_path = os.path.join(exp_dir, 'cae', config_cae_path(conf['cae']))
    elif conf['main_args']['stage'] == 2:
        conf['tcn'].update({'cae_path': os.path.join(exp_dir, 'cae', config_cae_path(conf['cae']))})
        model = CAE_DANet(conf['tcn'])
        model_path = os.path.join(exp_dir, model_dir)
        if conf['loss_fn']['spk_ce'] > 0:
            model.danet.add_softmax(output_size=len(SPKID), normalize=False)
            loss_fn['spk_ce'] = model.danet.spk_softmax
    else:
        raise ValueError('Training stage should be either 1 or 2!')
    model = torch.nn.DataParallel(model).cuda()
    opt = torch.optim.Adam(model.module.parameters(), lr=conf['optim']['lr'])

    # Validation metric
    metric_name = 'SISDRi'
    if metric_name == 'SISDRi':
        SISDRi = PermInvariantSISDR(backward_loss=False, improvement=True,
                                    return_individual_results=True)

    # Save config
    # os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    conf_path = os.path.join(model_path, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)

    # Train model
    tr_step = 0
    val_step = 0
    new_lr = conf['optim']['lr']
    halving = False
    best_val_loss = float("-inf")
    val_no_impv = 0
    for i in range(conf['training']['epochs']):
        metric_dic = {'train_{}'.format(metric_name): 0., 'val_{}'.format(metric_name): 0.}
        print("Training stage {} || Epoch: {}/{}".format(conf['main_args']['stage'],
                                                         i + 1,
                                                         conf['training']['epochs']))
        model.train()
        train_metric_mean = []
        for data_set in zip(tqdm(train_gens[0], desc='Training'), train_gens[1]) if len(train_gens) == 2 \
                else tqdm(train_gens[0], desc='Training'):  # mini-batch
            if not isinstance(data_set, tuple):
                data_set = (data_set,)
            for data in data_set:
                opt.zero_grad()
                m1wavs = data[0].unsqueeze(1).cuda()
                clean_wavs = data[-1].cuda()
                speaker_id = data[1]

                if conf['main_args']['stage'] == 1:
                    recon_sources, enc_masks, enc_mixture = model.module(m1wavs, clean_wavs)
                if conf['main_args']['stage'] == 2:
                    estimated_masks, enc_masks, enc_mixture, Wx, phase = model(m1wavs, clean_wavs, train=True,
                                                                               n_sources=clean_wavs.shape[1])
                    V = estimated_masks[1]  # V (B, K, F*T),  enc_masks (B, C, F*T)
                    A = estimated_masks[2]  # (B, nspk, K)
                    estimated_masks = estimated_masks[0] # estimated_masks (B, nspk, F*T)
                    recon_sources = model.module.get_rec_sources(
                        estimated_masks.view(m1wavs.shape[0], estimated_masks.shape[1], model.module.input_dim,-1),
                        enc_mixture, phase=phase)  # recovered waveform

                l_dict = dict()
                if conf['loss_fn']['sisdr'] > 0:
                    l_sisdr = loss_fn['sisdr'](recon_sources, clean_wavs).mean()
                    l_dict.update({'sisdr': conf['loss_fn']['sisdr'] * l_sisdr})
                if conf['loss_fn']['compact'] > 0:
                    enc_mixture = enc_mixture.view(enc_mixture.shape[0], -1).unsqueeze(1)
                    w = -enc_mixture / torch.sum(enc_mixture, dim=[1, 2], keepdim=True)
                    enc_masks[enc_masks <= 0.5] = 0
                    An = F.normalize(A.detach(), dim=2)
                    l_va = w * enc_masks * (torch.bmm(An, F.normalize(V, dim=1)))
                    l_va = l_va.sum(dim=[1,2]).mean()
                    l_dict.update({'compact': conf['loss_fn']['compact'] * l_va})
                if conf['loss_fn']['spk_circle'] > 0:
                    L = torch.zeros(A.shape[0], A.shape[1]).cuda()
                    for j in range(A.shape[0]):
                        for k in range(A.shape[1]):
                            L[j][k] = SPKID.index(speaker_id[k][j])
                    inp_sp, inp_sn = convert_label_to_similarity(A.view(-1, A.shape[2]), L.view(-1))
                    l_c = loss_fn['spk_circle'](inp_sp, inp_sn)
                    l_dict.update({'circle': conf['loss_fn']['spk_circle'] * l_c})
                if conf['loss_fn']['spk_ce'] > 0:
                    label = torch.zeros(A.shape[0], A.shape[1], dtype=torch.int64).cuda()
                    for j in range(A.shape[0]):
                        for k in range(A.shape[1]):
                            label[j][k] = SPKID.index(speaker_id[k][j])
                    if conf['tcn']['sim'] == 'cos':
                        l_softmax = loss_fn['spk_ce'](F.normalize(A.view(-1, A.shape[2]), p=2, dim=1), label.view(-1))
                    else:
                        l_softmax = loss_fn['spk_ce'](A.view(-1, A.shape[2]), label.view(-1))
                    l_dict.update({'spksoftmax':conf['loss_fn']['spk_ce'] * l_softmax})

                # Loss back-propagation
                l = torch.tensor(0.0).cuda()
                for loss in l_dict.values():
                    if not math.isinf(loss) and not math.isnan(loss):
                        l = l + loss

                if not conf['cae']['stft'] or conf['main_args']['stage'] == 2:
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    opt.step()

                train_metric = SISDRi(recon_sources, clean_wavs, initial_mixtures=m1wavs)
                train_metric_mean += train_metric.tolist()

        train_metric_mean = np.mean(train_metric_mean)
        metric_dic['train_{}'.format(metric_name)] = train_metric_mean
        tr_step += 1

        if val_gens is not None:
            model.eval()
            with torch.no_grad():
                val_metric_mean = []
                for data_set in zip(tqdm(val_gens[0], desc='Validation'), val_gens[1]) if len(val_gens) == 2 \
                        else tqdm(val_gens[0], desc='Validation'):  # mini-batch
                    if not isinstance(data_set, tuple):
                        data_set = (data_set,)
                    for data in data_set:
                        m1wavs = data[0].unsqueeze(1).cuda()
                        clean_wavs = data[-1].cuda()

                        if conf['main_args']['stage'] == 1:
                            recon_sources, _, _ = model.module(m1wavs, clean_wavs)
                        if conf['main_args']['stage'] == 2:
                            estimated_masks, _, enc_mixture, _, phase = model(m1wavs, clean_wavs, train=True,
                                                                                       n_sources=clean_wavs.shape[1])
                            V = estimated_masks[1]  # V (B, K, F*T),  enc_masks (B, C, F*T)
                            A = estimated_masks[2]  # (B, nspk, K)
                            estimated_masks = estimated_masks[0]  # estimated_masks (B, nspk, F*T)
                            recon_sources = model.module.get_rec_sources(
                                estimated_masks.view(m1wavs.shape[0], estimated_masks.shape[1], model.module.input_dim,
                                                     -1),
                                enc_mixture, phase=phase)  # recovered waveform

                        val_metric = SISDRi(recon_sources, clean_wavs, initial_mixtures=m1wavs)
                        val_metric_mean += val_metric.tolist()
            val_metric_mean = np.mean(val_metric_mean)
            metric_dic['val_{}'.format(metric_name)] = val_metric_mean
            val_step += 1


        # Adjust learning rate (halving)
        if conf['training']['half_lr']:
            val_loss = round(val_metric_mean, 2)  # keep two decimal places
            if val_loss <= best_val_loss:
                val_no_impv += 1
                if val_no_impv % 6 == 0:
                    halving = True
                if val_no_impv >= 20 and conf['training']['early_stop']:
                    print("No imporvement for 20 epochs, early stopping.")
                    break
            else:
                best_val_loss = val_loss
                val_no_impv = 0
            if halving:
                optim_state = opt.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                opt.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                halving = False
                # val_no_impv = 0

        CAE.save_if_best(save_dir=model_path,
                         model=model.module,
                         optimizer=opt,
                         epoch=tr_step,
                         tr_loss=train_metric_mean,
                         cv_loss=val_metric_mean,
                         cv_loss_name='SISDRi',
                         save_every=50)
        pprint(metric_dic)


if __name__ == '__main__':
    with open('conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
