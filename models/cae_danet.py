from models import cae
from models.tcn import TCN
from models.model_base import Model_base
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
from losses.spk_loss import SoftmaxLoss


class CAE_DANet(Model_base):
    def __init__(self, model_args):
        super(CAE_DANet, self).__init__()
        self.model_args = model_args
        # CAE
        self.cae_dir = model_args['cae_path']  # cae_dir_path
        # Try to load the pre-trained CAE
        try:
            self.cae = cae.CAE.load_model(self.cae_dir, model_state='best')
        except Exception as e:
            print(e)
            raise ValueError("Could not load best pretrained adaptive "
                             "front-end from: {} :(".format(self.cae_dir))
        self.encoder = self.cae.encoder
        self.decoder = self.cae.decoder

        self.freeze_cae()

        # DANet
        self.input_dim = self.encoder.conv.weight.data.shape[0]
        self.K = model_args['K']
        self.act_fn = model_args['act_fn']
        self.da_sim = model_args['sim']
        self.alpha = model_args['alpha']
        self.weight = model_args['weight']
        self.v_act = model_args['v_act']
        self.v_norm = model_args['v_norm']

        # TCN
        self.B = model_args['B']
        self.H = model_args['H']
        self.P = model_args['P']
        self.X = model_args['X']
        self.R = model_args['R']
        if 'Sc' not in model_args.keys():
            self.Sc = model_args['B']

        self.net = 'TCN'
        if self.net == 'TCN':
            self.danet = TCN_DANet(input_dim=self.input_dim,
                                   embed_dim=self.K,
                                   B=self.B,
                                   H=self.H,
                                   X=self.X,
                                   R=self.R,
                                   P=self.P,
                                   Sc=self.Sc,
                                   act_fn=self.act_fn,
                                   da_sim=self.da_sim,
                                   alpha=self.alpha,
                                   V_activate=self.v_act,
                                   V_norm=self.v_norm)

        self.eps = 10e-9

    def freeze_cae(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, mixture, clean_sources, train=True, n_sources=None, A=None):
        # mixture: (B, 1, seg_len)
        # clean_sources: (B, C, seg_len)
        enc_mixture = self.encoder(mixture)  # (B, F, T)
        if self.cae.use_stft:
            phase = enc_mixture[1]
            enc_mixture = enc_mixture[0]
        else:
            phase = None
        if clean_sources is not None:
            enc_masks = self.get_target_masks(clean_sources)  # (B, C, F, T) .cae
            enc_masks_lstm = None
            if self.net == 'LSTM':
                enc_masks_lstm = torch.transpose(enc_masks, 2, 3)  # (B, C, T, F)
                enc_masks_lstm = enc_masks_lstm.contiguous().view(enc_masks_lstm.shape[0], enc_masks_lstm.shape[1],
                                                                  -1)  # (B, C，T*F)
            enc_masks = enc_masks.contiguous().view(enc_masks.shape[0], enc_masks.shape[1], -1)  # (B, C，F*T)
        else:
            enc_masks = 1
            enc_masks_lstm = None
        rec_masks = None
        weight = None
        if self.weight == 'Wthr':
            weight = self.get_enc_weight(enc_mixture, thr=0.01).cuda()  # (B, 1, F*T) .cae
        elif self.weight == 'Wmr':
            weight = enc_mixture/torch.sum(enc_mixture, dim=[1,2], keepdim=True)  # (B, F, T)
            # weight = enc_mixture / enc_mixture.max()  # (B, F, T), this is equivalent to the equation above
            weight = weight.view(enc_mixture.shape[0],-1).unsqueeze(1)  # (B, 1, F*T)
        elif self.weight == 'Wthrp':
            weight = self.get_enc_weight(enc_mixture, thr_p=0.9).cuda()  # (B, 1, F*T) .cae
        else:
            weight = 1
        if self.net == 'TCN' or self.net == 'DPRNN':
            # enc_mixture = enc_mixture*weight.squeeze(1).view(weight.shape[0],enc_mixture.shape[1],-1)
            rec_masks = self.danet(enc_mixture, enc_masks, weight, train=train, n_sources=n_sources, A=A)  # (B, C, F*T)
            # rec_masks = rec_masks*weight   # 12.21
            # enc_masks = enc_masks*weight   # 12.21
            # When train=True, rec_masks is rec_masks (B, C, F*T), otherwise rec_masks is embedding V (B, K, F*T)
        elif self.net == 'LSTM':
            # Transpose to adapt the DANet
            enc_mixture_tmp = torch.transpose(enc_mixture, 1, 2)  # (B, T, F)
            rec_masks = self.danet(enc_mixture_tmp, enc_masks, weight, train=train, n_sources=n_sources, A=A)  # (B, C, T*F)
        else:
            raise ValueError('Embedding net should be TCN, DPRNN, or LSTM')

        return rec_masks, enc_masks, enc_mixture, weight, phase

    def get_rec_sources(self, enc_masks, enc_mixture, U=None, phase=None):
        # enc_masks.shape = (B, C, F, T)
        # enc_mixture.shape = (B, F, T)
        est_sources = self.cae.get_rec_sources(enc_masks, enc_mixture, phase)
        return est_sources

    def get_target_masks(self, clean_sources):
        enc_masks = self.cae.get_target_masks(clean_sources)
        return enc_masks

    def get_encoded_sources(self, mixture, clean_sources):
        enc_mixture = self.encoder(mixture)
        enc_masks = self.get_target_masks(clean_sources)
        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        return s_recon_enc

    def get_enc_weight(self, enc_mixture, thr_p=0.9, thr=None):
        # input (B, F, T)
        # output (B, 1, F*T)
        enc_mixture = enc_mixture.contiguous().view(enc_mixture.shape[0], -1)  # input (B, F*T)
        weight = torch.zeros(enc_mixture.shape)

        if thr:
            weight[enc_mixture>=thr] = 1
            return weight.unsqueeze(1)
        else:
            _, ind = torch.topk(enc_mixture, int(np.round(thr_p * enc_mixture.shape[1])))
            for i in range(weight.shape[0]):
                weight[i, ind[i]] = 1
            return weight.unsqueeze(1)


class kMeansIter(nn.Module):
    def __init__(self, type='soft', alpha=5.0, dist_type='negL2norm', anchor=0, K=16):
        super(kMeansIter, self).__init__()
        self.type = type
        self.alpha = alpha
        self.dist_type = dist_type
        # if anchor>0:
            # self.anchor = nn.Parameter(torch.Tensor(anchor, K))

    def forward(self, V, U, weight=1):
        # V: (B, F*T, K)   U: (B, nspk, K)

        # soft k-means
        if self.type == 'soft':
            Y = torch.softmax(self.distance(V, U, self.alpha, self.dist_type), dim=1)

        # hard k-means
        if self.type == 'hard':
            Y = F.one_hot(torch.argmax(self.distance(V, U, self.alpha, self.dist_type), dim=1)).transpose(1, 2)

        Y = Y*weight
        V_Y = torch.bmm(Y, V)  # B, nspk, K
        sum_Y = torch.sum(Y, 2, keepdim=True).expand_as(V_Y)  # B, nspk, K
        U_out = V_Y / (sum_Y + 1e-9)  # B, nspk, K
        return Y, U_out

    @classmethod
    def distance(cls, V, U, alpha=5.0, dist_type='negL2norm'):
        # V: (B, F*T, K)   U: (B, nspk, K)
        if dist_type == 'negL2norm':
            dist = -alpha * torch.norm(V - U[:, 0, :].unsqueeze(1), p=2, dim=2).unsqueeze(1)  # (B, 1, F*T)
            for j in range(U.shape[1]-1):
                temp = -alpha * torch.norm(V - U[:, j+1, :].unsqueeze(1), p=2, dim=2).unsqueeze(1)
                dist = torch.cat((dist, temp), dim=1)
        elif dist_type == 'dotProduct':
            dist = torch.bmm(U, V.permute(0, 2, 1))  # B, nspk, F*T
        elif dist_type == 'cos':
            dist = alpha*torch.bmm(F.normalize(U, dim=2, p=2), F.normalize(V.permute(0, 2, 1), dim=1, p=2))  # B, nspk, F*T
        else:
            raise ValueError("dist_type must be negL2norm or dotProduct or cos.")
        return dist  # (B, nspk, F*T)


# base module for FaSNet
class DANet(nn.Module):
    def __init__(self, embed_dim=20, act_fn='softmax', da_sim='dotProduct', alpha=5.0, V_activate=True,
                 V_norm=False, A_norm=False, A_mask=False, dist_scaler=0, n_sources=2):
        super(DANet, self).__init__()

        # parameters
        self.embed_dim = embed_dim
        self.act_fn = act_fn
        self.da_sim = da_sim

        self.eps = 1e-8
        self.n_sources = n_sources

        self.kmeans_type = None
        self.kmeans_layers = None
        self.kmeans_dist = None
        self.n_init = None  # randomly initiate A for n_init times, choose A with the lowest error
        self.embedding_net = None
        self.spk_softmax = None

        self.V_activate = V_activate
        self.V_norm = V_norm
        self.alpha = alpha
        self.A_norm = A_norm
        self.A_mask = A_mask
        self.dist_scaler = dist_scaler

        if self.dist_scaler != 0:
            if self.dist_scaler<0:
                self.scaler = nn.Parameter(torch.Tensor(1))
                self.scaler.data.fill_(-dist_scaler)
            else:
                self.scaler = self.dist_scaler

        self.km = None  # k-means used in test stage

    def add_kmeans(self, kmeans_type='soft', alpha=None, iter=10, dist_type='negL2norm', n_init=1):
        # Call this function to add k-means layers and fine-tune
        self.kmeans_type = kmeans_type
        self.n_init = n_init
        self.kmeans_dist = dist_type
        if self.kmeans_type:
            if alpha is None:
                alpha = self.alpha
            self.kmeans_layers = nn.ModuleList([])
            for i in range(iter):
                self.kmeans_layers.append(kMeansIter(kmeans_type, alpha, dist_type))

    def add_softmax(self, output_size, normalize=True, W=None):
        print('softmax_norm:', normalize)
        self.spk_softmax = SoftmaxLoss(self.embed_dim, output_size=output_size, normalize=normalize, W=W)

    def forward(self, input, mask=None, weight=None, train=True, n_sources=None, V=None, A=None):
        """
        input: shape (B, F, T)
        """
        if V is None:
            V = self.embedding_net(input)  # (B, F*K, T)
        if self.V_activate:
            V = torch.tanh(V)
        V = V.contiguous().view(V.shape[0], self.embed_dim, -1, V.shape[2])  # (B, K, F, T)
        V = V.contiguous().view(V.shape[0], self.embed_dim, -1)  # (B, K, F*T)
        VT = torch.transpose(V, 1, 2)  # (B, F*T, K)
        if self.A_mask:
            mask[mask<self.A_mask] = 0.0
            mask[mask>=self.A_mask] = 1.0
        if self.V_norm:
            # V = F.normalize(V, p=2, dim=1)
            VT = F.normalize(VT, p=2, dim=2)
        if train:
            # calculate the ideal attractors
            # first calculate the source assignment matrix Y
            if A is None:
                if self.kmeans_type:
                    if not n_sources:
                        n_sources = self.n_sources
                    # Y = torch.zeros(V.shape[0], n_sources, V.shape[2]).cuda()  # (B, nspk, FT)
                    A = torch.zeros(V.shape[0], n_sources, self.embed_dim).cuda()  # (B, nspk, K)
                    temp_A = torch.zeros(V.shape[0], n_sources, self.embed_dim).cuda()  # (B, nspk, K)

                    err = torch.zeros(V.shape[0])
                    for j in range(self.n_init):
                        for i in range(V.shape[0]):
                            ind = torch.randperm(V.shape[2])[:n_sources].cuda()
                            temp_A[i] = VT[i, ind, :]  # randomly initiate A
                        # print('kmeans++++++++++++')
                        for i in range(len(self.kmeans_layers)):
                            Y, temp_A = self.kmeans_layers[i](VT, temp_A, weight)
                        if self.n_init == 1:
                            A = temp_A
                        else:
                            dist = -kMeansIter.distance(VT, temp_A, alpha=self.alpha, dist_type=self.kmeans_dist)  # B, nspk, F*T
                            temp_err = torch.sum(dist * Y, dim=[1, 2])
                            for i in range(A.shape[0]):
                                if j == 0 or temp_err[i] < err[i]:
                                    err[i] = temp_err[i]
                                    if A[i].shape[0] != temp_A[i].shape[0]:
                                        print(ind)
                                    A[i] = temp_A[i]
                else:
                    Y = mask * weight  # B, nspk，F*T
                    # attractors are the weighted average of the embeddings
                    # calculated by V and Y
                    V_Y = torch.bmm(Y, VT)  # B, nspk, K
                    sum_Y = torch.sum(Y, 2, keepdim=True).expand_as(V_Y)  # B, nspk, K
                    A = V_Y / (sum_Y + self.eps)  # B, nspk, K
        else:
            if not n_sources:
                n_sources = self.n_sources
            A = torch.zeros(V.shape[0], n_sources, V.shape[1])
            if self.km is None:
                if self.da_sim == 'cos':
                    self.km = SphericalKMeans(n_clusters=n_sources) #, init='k-means++', n_init=5, max_iter=20)
                if self.da_sim == 'dotProduct' or self.da_sim == 'negL2norm':
                    self.km = KMeans(n_clusters=n_sources) # , init='k-means++', n_init=5, max_iter=20)
            for i in range(VT.shape[0]):
                # skm.fit(VT[i].cpu())
                self.km.fit(VT[i].cpu(), sample_weight=weight[i][0].cpu())
                A[i] = torch.from_numpy(self.km.cluster_centers_)
            A = A.cuda()
            # return V  # (B, K, F*T)

        # calculate the similarity between embeddings and attractors
        if self.A_norm:
            A = F.normalize(A, p=2, dim=2)
        if self.da_sim == 'dotProduct':
            dist = torch.bmm(A, V)  # B, nspk, F*T
        elif self.da_sim == 'negL2norm':
            dist = kMeansIter.distance(VT, A, alpha=self.alpha, dist_type='negL2norm')
        elif self.da_sim == 'cos':
            dist = kMeansIter.distance(VT, A, alpha=self.alpha, dist_type='cos')  # B, nspk, F*T
        elif self.da_sim == 'sin':
            sin_va = F.relu(1 - (A.bmm(VT.transpose(1, 2))).pow(2)).sqrt()  # B, nspk, TF
            # sin_va = torch.norm(torch.cross(A, VT, dim=2))
            if A.shape[1] == 2:
                mask = sin_va[:, [1, 0], :] / (torch.sum(sin_va, dim=1, keepdim=True) + self.eps)
            if A.shape[1] > 2:
                mask = torch.zeros(mask.shape).cuda()
                ind = torch.arange(mask.shape[1])
                for i in range(mask.shape[1]):
                    mask[:, i, :] = sin_va[:, ind[ind != i], :].min(dim=1).values / \
                                    (sin_va[:, i, :] + sin_va[:, ind[ind != i], :].min(dim=1).values + self.eps)
            return (mask, V, A)
        else:
            raise ValueError('Attractor similarity must be dotProduct or negL2norm or cos')
        # re-scale the similarity distance
        if self.dist_scaler:
            dist = dist * self.scaler
        # generate the masks
        if self.act_fn == 'softmax':
            mask = F.softmax(dist, dim=1)  # B, nspk, F*T
            return (mask, V, A)
        elif self.act_fn == 'sigmoid':
            mask = torch.sigmoid(dist)  # B, nspk, F*T
            return (mask, V, A)
        else:
            raise ValueError('Activation function must be softmax or sigmoid')



class TCN_DANet(DANet):
    def __init__(self, input_dim=129, embed_dim=20, B=128, H=256, X=8, R=3, P=3, skip=True,
                 causal=False, dilated=True, pad_layer=None, Sc=None, act_fn='softmax',
                 da_sim='dotProduct', alpha=10.0, V_activate=True, V_norm=False,
                 dist_scaler=0, n_sources=2):
        super(TCN_DANet, self).__init__(embed_dim=embed_dim, act_fn=act_fn, da_sim=da_sim,
                                        alpha=alpha, V_activate=V_activate, V_norm=V_norm,
                                        dist_scaler=dist_scaler, n_sources=n_sources)

        self.input_dim = input_dim
        self.B = B
        self.H = H
        self.X = X
        self.R = R
        self.P = P
        self.Sc =Sc
        self.skip = skip
        self.causal = causal
        self.dilated = dilated
        self.pad_layer = pad_layer

        self.embedding_net = TCN(input_dim=self.input_dim, output_dim=self.input_dim*self.embed_dim,
                       BN_dim=self.B, hidden_dim=self.H, layer=self.X, stack=self.R, kernel=self.P,
                       skip=self.skip, causal=self.causal, dilated=self.dilated, pad_layer=self.pad_layer, Sc=self.Sc)



