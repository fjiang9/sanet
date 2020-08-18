import os
import argparse
import json
import torch
import torch.nn.functional as F
from models.cae import CAE
from models.cae_danet import CAE_DANet
from data import LibriMix, LIBRISPEECH_SPKID, TEST
from torch.utils.data import DataLoader
from losses.sisdr import PermInvariantSISDR
from losses.spk_loss import CircleLoss, convert_label_to_similarity
from tqdm import tqdm
import yaml
from pprint import pprint
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
import numpy as np
import math
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')
parser.add_argument('--stage', type=int, default=2,
                    help='stage1: cae pre-training, stage2: embedding net training/testing')
parser.add_argument('--model_dir', default='sanet')
parser.add_argument('--cuda', type=str, nargs="+", default=['0'])
compute_metrics = ['si_sdr']

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
    test_gens = dict()
    for dataset_name, info in TEST.items():
        test_set = LibriMix(csv_path=info[1],
                           sample_rate=conf['data']['sample_rate'],
                           n_src=info[0],
                           segment=None)
        test_gen = DataLoader(test_set, shuffle=False,
                              batch_size=1,
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)
        test_gens.update({dataset_name: test_gen})
    SPKID = LIBRISPEECH_SPKID['test']

    # Define model, optimizer + scheduler
    # if conf['main_args']['stage'] == 1:
    #     model = CAE(conf['cae'])
    #     model_path = os.path.join(exp_dir, 'cae', config_cae_path(conf['cae']))
    # elif conf['main_args']['stage'] == 2:
    model_path = os.path.join(exp_dir, model_dir)
    model = CAE_DANet.load_model(model_path, model_state='best')

    # else:
    #     raise ValueError('Training stage should be either 1 or 2!')
    model = torch.nn.DataParallel(model).cuda()

    # Used to reorder sources only
    # loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    loss_func = PermInvariantSISDR(backward_loss=False, improvement=True,
                                   return_individual_results=True, pit=True)


    # Test
    model.eval()
    model.module.danet.add_kmeans(kmeans_type='hard', alpha=10, iter=20, dist_type='cos', n_init=5)
    with torch.no_grad():
        for set_name, test_gen in test_gens.items():  # different test sets
            series_list = []
            spk_centroids = dict()
            for data in tqdm(test_gen, desc='Testing {}'.format(set_name), ncols=100):
                m1wavs = data[0].unsqueeze(1).cuda()
                clean_wavs = data[-1].cuda()
                speaker_id = data[1]

                if conf['main_args']['stage'] == 1:
                    recon_sources, _, _ = model.module(m1wavs, clean_wavs)
                    reordered_sources = recon_sources
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

                    # loss, reordered_sources = loss_func(recon_sources, clean_wavs, return_est=True)
                    test_sisdri = loss_func(recon_sources, clean_wavs, initial_mixtures=m1wavs)
                    reordered_sources = torch.zeros(recon_sources.shape).cuda()
                    for j in range(A.shape[0]):
                        reordered_sources[j] = recon_sources[:, loss_func.best_perm[j], :]
                        z = 0
                        for k in loss_func.best_perm[j]:
                            spk_id = speaker_id[z][j]
                            if spk_id in spk_centroids.keys():
                                spk_centroids[spk_id] = torch.cat(
                                    (spk_centroids[spk_id], A[j, k].unsqueeze(0).detach().cpu()),
                                    dim=0)
                            else:
                                spk_centroids[spk_id] = A[j, k].unsqueeze(0).detach().cpu()
                            z += 1

                m1wavs = m1wavs[:, :, :recon_sources.shape[2]]
                clean_wavs = clean_wavs[:, :, :recon_sources.shape[2]]

                mix_np = m1wavs.squeeze(0).cpu().data.numpy()
                sources_np = clean_wavs.squeeze(0).cpu().data.numpy()
                est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
                utt_metrics = get_metrics(mix_np, sources_np, est_sources_np,
                                          sample_rate=conf['data']['sample_rate'],
                                          metrics_list=compute_metrics)

                # utt_metrics['mix_path'] = test_set.mix[idx][0]
                series_list.append(pd.Series(utt_metrics))
                # pprint(utt_metrics)

            # Save all metrics to the experiment folder.
            all_metrics_df = pd.DataFrame(series_list)
            all_metrics_df.to_csv(os.path.join(model_path, '{}_all_metrics.csv'.format(set_name)))
            torch.save(spk_centroids, os.path.join(model_path, '{}_spkC.pt'.format(set_name)))

            # Print and save summary metrics
            final_results = {}
            for metric_name in compute_metrics:
                input_metric_name = 'input_' + metric_name
                ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
                final_results[metric_name] = all_metrics_df[metric_name].mean()
                final_results[metric_name + '_imp'] = ldf.mean()
            print('Overall metrics :')
            pprint(final_results)
            with open(os.path.join(model_path, '{}_final_metrics.json'.format(set_name)), 'w') as f:
                json.dump(final_results, f, indent=0)


if __name__ == '__main__':
    main_args = vars(parser.parse_args())
    with open(os.path.join(main_args['exp_dir'], main_args['model_dir'], 'conf.yml')) as f:
        arg_dic = yaml.safe_load(f)
    arg_dic['main_args'].update(main_args)
    main(arg_dic)
