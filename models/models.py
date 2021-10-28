import torch
from torch import nn
from torch.nn import functional as F
from models import base_model
from models import cca_loss

class DCCA(nn.Module):

    def __init__(self,
                 view1_size: int,
                 view2_size: int,
                 view1_hidden: list = None,
                 view2_hidden: list = None,
                 latent_dims: int = 2,
                 device: str = 'cuda'):
        super(DCCA, self).__init__()

        if view1_hidden is None:
            view1_hidden = [128]
        self.encoder_1 = base_model.Encoder(view1_hidden, view1_size, latent_dims).double()
        if view2_hidden is None:
            view2_hidden = [128]
        self.encoder_2 = base_model.Encoder(view2_hidden, view2_size, latent_dims).double()

        self.latent_dims = latent_dims
        self.device = device
        self.cca_objective = cca_loss.cca(self.latent_dims, device=self.device)

    def encode(self, x_1, x_2):
        z_1 = self.encoder_1(x_1)
        z_2 = self.encoder_2(x_2)
        return z_1, z_2

    def forward(self, x_1, x_2):
        z_1, z_2 = self.encode(x_1, x_2)
        return z_1, z_2

    def loss(self, z_1, z_2):
        cca_loss = self.cca_objective.loss(z_1, z_2)
        return cca_loss

class DCCAE(nn.Module):

    def __init__(self,
                 view1_size: int,
                 view2_size: int,
                 view1_hidden: list = None,
                 view2_hidden: list = None,
                 latent_dims: int = 2,
                 device: str = 'cuda',
                 lam=0):
        super(DCCAE, self).__init__()

        if view1_hidden is None:
            view1_hidden = [128]
        self.encoder_1 = base_model.Encoder(view1_hidden, view1_size, latent_dims).double()
        self.decoder_1 = base_model.Decoder(view1_hidden, latent_dims, view1_size).double()
        if view2_hidden is None:
            view2_hidden = [128]
        self.encoder_2 = base_model.Encoder(view2_hidden, view2_size, latent_dims).double()
        self.decoder_2 = base_model.Decoder(view2_hidden, latent_dims, view2_size).double()

        self.latent_dims = latent_dims
        self.device = device
        self.cca_objective = cca_loss.cca(self.latent_dims, device=self.device)
        self.lam = lam

    def encode(self, x_1, x_2):
        z_1 = self.encoder_1(x_1)
        z_2 = self.encoder_2(x_2)
        return z_1, z_2

    def decode(self, z_1, z_2):
        x_1_recon = self.decoder_1(z_1)
        x_2_recon = self.decoder_2(z_2)
        return x_1_recon, x_2_recon

    def forward(self, x_1, x_2):
        z_1, z_2 = self.encode(x_1, x_2)
        x_1_recon, x_2_recon = self.decode(z_1, z_2)
        return z_1, z_2, x_1_recon, x_2_recon

    def loss(self, x_1, x_2, z_1, z_2, x_1_recon, x_2_recon):
        recon_1 = F.mse_loss(x_1_recon, x_1, reduction='sum')
        recon_2 = F.mse_loss(x_2_recon, x_2, reduction='sum')
        recon_loss = self.lam * recon_1 + self.lam * recon_2
        cca_loss = self.cca_objective.loss(z_1, z_2)
        return recon_loss + cca_loss, recon_loss, cca_loss

class FR_DCCA(nn.Module):

    def __init__(self,
                 view1_size: int,
                 view2_size: int,
                 view1_hidden: list = None,
                 view2_hidden: list = None,
                 latent_dims: int = 2,
                 device: str = 'cuda',
                 recon_p=1e-4,
                 cca_p=1,
                 js_p=1):
        super(FR_DCCA, self).__init__()


        if view1_hidden is None:
            view1_hidden = [128]
        self.encoder_1_shared = base_model.Encoder(view1_hidden, view1_size, latent_dims).double()
        self.encoder_1_specific = base_model.Encoder(view1_hidden, view1_size, latent_dims).double()
        self.decoder_1 = base_model.Decoder(view1_hidden, latent_dims * 2, view1_size).double()

        if view2_hidden is None:
            view2_hidden = [128]
        self.encoder_2_shared = base_model.Encoder(view2_hidden, view2_size, latent_dims).double()
        self.encoder_2_specific = base_model.Encoder(view2_hidden, view2_size, latent_dims).double()
        self.decoder_2 = base_model.Decoder(view2_hidden, latent_dims * 2, view2_size).double()

        self.latent_dims = latent_dims
        self.device = device
        self.recon_p = recon_p
        self.cca_p = cca_p
        self.js_p = js_p
        self.cca_objective = cca_loss.cca(self.latent_dims, self.device)


    def init_param(self):
        self.encoder_1_specific.init_encoder()
        self.encoder_2_specific.init_encoder()
        self.encoder_1_shared.init_encoder()
        self.encoder_2_shared.init_encoder()
        self.decoder_1.init_decoder()
        self.decoder_2.init_decoder()

    def encode(self, x_1, x_2):
        view1_specific = self.encoder_1_specific(x_1)
        view1_shared = self.encoder_1_shared(x_1)
        view2_specific = self.encoder_2_specific(x_2)
        view2_shared = self.encoder_2_shared(x_2)
        return view1_specific, view1_shared, view2_specific, view2_shared

    def decode(self, view1_specific, view1_shared, view2_specific, view2_shared):
        view1_recon = self.decoder_1(torch.cat([view1_specific, view1_shared], dim=1))
        view2_recon = self.decoder_2(torch.cat([view2_specific, view2_shared], dim=1))
        return view1_recon, view2_recon

    def forward(self, x_1, x_2):
        view1_specific, view1_shared, view2_specific, view2_shared = self.encode(x_1, x_2)
        view1_recon, view2_recon = \
            self.decode(view1_specific, view1_shared, view2_specific, view2_shared)
        return view1_specific, view1_shared, \
               view2_specific, view2_shared, \
               view1_recon, \
               view2_recon

    def orthogonal_loss(self, shared, specific):
        shared = torch.sigmoid(shared)
        specific = torch.sigmoid(specific)
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = shared.mul(specific)
        cost = correlation_matrix.mean()
        return cost

    def kl_loss(self, shared, specific):
        shared = torch.sigmoid(shared)
        specific = torch.sigmoid(specific)
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        #  share guide specific
        p_shared = F.softmax(shared, dim=-1)
        log_p_specific = F.log_softmax(specific, dim=-1)
        kl_cost = F.kl_div(log_p_specific, p_shared, reduction='batchmean')
        return kl_cost

    def js_loss(self, shared, specific):
        M = (shared + specific) / 2
        kl_1 = self.kl_loss(shared, M)
        kl_2 = self.kl_loss(specific, M)
        return 0.5*kl_1 + 0.5*kl_2

    def loss(self, x_1, x_2, view1_specific, view1_shared, view2_specific, view2_shared,
             view1_recon, view2_recon):
        view1_recon_loss = F.mse_loss(view1_recon, x_1, reduction='sum')
        view2_recon_loss = F.mse_loss(view2_recon, x_2, reduction='sum')

        cca_loss = self.cca_objective.loss(view1_shared, view2_shared)
        js_loss = self.js_loss(view1_shared, view1_specific) + \
                  self.orthogonal_loss(view2_shared, view2_specific)
        recon_loss = view1_recon_loss + view2_recon_loss
        return recon_loss*self.recon_p + cca_loss*self.cca_p + js_loss*self.js_p, \
               recon_loss*self.recon_p, \
               cca_loss*self.cca_p






