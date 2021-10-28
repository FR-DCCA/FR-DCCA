import random
import copy
import time
import sklearn
import torch
from sklearn.cluster import SpectralClustering
from torch import optim
from torch.utils.data import TensorDataset
import numpy as np
from models import models
from read_data_validation import MNISTValidation
from evaluation import match_new_predict
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class Wrapper:

    def __init__(self,
                 latent_dims: int = 2,
                 learning_rate=1e-3,
                 epoch_num: int = 1,
                 batch_size: int = 16,
                 recon_p=1e-4,
                 cca_p=1,
                 js_p=1,
                 patience: int = 10,
                 view1_hidden: list = None,
                 view2_hidden: list = None):
        self.latent_dims = latent_dims
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.device = torch.device("cuda:1")
        self.patience = patience
        self.batch_size = batch_size
        self.view1_hidden = view1_hidden
        self.view2_hidden = view2_hidden
        self.recon_p = recon_p
        self.cca_p = cca_p
        self.js_p = js_p


    def fit(self):
        # set seed
        self.seed_torch(seed=24)
        # load data
        train = MNISTValidation(set_name='train')
        val = MNISTValidation(set_name='val')
        test = MNISTValidation(set_name='test')
        # get feature size
        view1_train = train.train_page_data
        view2_train = train.train_link_data

        # Data Loader for easy mini-batch return in training, set num workers
        train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=self.batch_size,
                                                   num_workers=5, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=self.batch_size,
                                                  num_workers=5, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=self.batch_size,
                                                 num_workers=5, shuffle=True)

        # model: fr_dcca
        self.model = models.FR_DCCA(
            view1_size=view1_train.shape[-1],
            view2_size=view2_train.shape[-1],
            view1_hidden=self.view1_hidden,
            view2_hidden=self.view2_hidden,
            latent_dims=self.latent_dims,
            device=self.device,
            recon_p=self.recon_p,
            cca_p=self.cca_p,
            js_p=self.js_p)

        # config cluster
        self.cluster = SpectralClustering(n_clusters=10,
                                          n_neighbors=5,
                                          n_init=5,
                                          n_jobs=-1,
                                          random_state=24)

        model_params = sum(p.numel() for p in self.model.parameters())
        best_model = copy.deepcopy(self.model.state_dict())
        # print(self.model)
        print("Number of model parameters {}".format(model_params))
        self.model.double().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        min_val_loss = 1e5
        epochs_no_improve = 0  # early stop count
        early_stop = False   # early stop flag

        for epoch in range(self.epoch_num):
            if early_stop == False:
                epoch_train_loss = self.train_epoch(train_loader)
                epoch_val_loss = self.val_epoch(val_loader)
                print('====> Epoch: {} train loss: {:.4f} val loss: '
                      '{:.4f}'.format(
                        epoch, epoch_train_loss, epoch_val_loss))

                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    print('min_val_loss %0.4f' % min_val_loss)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == self.patience:
                        print('Early stopping!')
                        early_stop = True
                        self.model.load_state_dict(best_model)

        # best val model to test num times and get mean acc, nmi
        print("-------------------------test--------------------------")
        test_acc = 0.0
        test_nmi = 0.0
        test_num = 1
        for i in range(test_num):
            test_acc_, test_nmi_ = self.test_cluster(test_dataloader=test_loader)
            test_acc += test_acc_
            test_nmi += test_nmi_
        print("test 10000 sample one time acc, nmi :", test_acc / test_num, test_nmi / test_num)
        return self

    def seed_torch(self, seed=24):
        # random seed
        random.seed(seed)
        # torch cpu model seed
        torch.manual_seed(seed)
        # torch gpu model seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # multi-gpu seed
        # numpy random seed
        np.random.seed(seed)

    def train_epoch(self, train_dataloader: torch.utils.data.DataLoader):
        self.model.train()
        train_loss = 0
        for batch_idx, (x, y, label) in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            x = x.double()
            y = y.double()
            x, y = x.to(self.device), y.to(self.device)
            model_outputs = self.model(x, y)
            # loss_recon and loss_cca used for adjust
            loss_all, loss_recon, loss_cca = self.model.loss(x, y, *model_outputs)
            train_loss += loss_all.item()
            loss_all.backward()
            self.optimizer.step()
        return train_loss / len(train_dataloader)

    def val_epoch(self, val_dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y, val_label) in enumerate(val_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                x = x.double()
                y = y.double()
                model_outputs = self.model(x, y)
                loss_all, loss_recon, loss_cca = self.model.loss(x, y, *model_outputs)
                val_loss += loss_all.item()
        return val_loss / len(val_dataloader)

    # use the best model to model the test data,
    # and use cluster to cluster (test data) one time,
    def test_cluster(self, test_dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        test_view1 = np.empty(shape=(0, self.latent_dims*2))  # use only view1 model result to classify
        test_label = np.empty(shape=0)
        # for batch get the all model result -> test data
        for batch_idx, (x, y, label) in enumerate(test_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.double()
            y = y.double()
            model_outputs = self.model(x, y)
            view1_specific = model_outputs[0]
            view1_shared = model_outputs[1]
            view1 = torch.cat([view1_specific, view1_shared], dim=1)
            view1 = view1.cpu().data.numpy()
            label = label.cpu().data.numpy()
            test_view1 = np.concatenate([test_view1, view1], axis=0)
            test_label = np.concatenate([test_label, label], axis=0)

        predict = self.cluster.fit_predict(test_view1)
        new_predict = match_new_predict(y_true=test_label, y_pred=predict)
        acc = sklearn.metrics.accuracy_score(test_label, new_predict)
        nmi = sklearn.metrics.cluster.normalized_mutual_info_score(test_label, new_predict, average_method='arithmetic')
        return acc, nmi


if __name__ == '__main__':

    # The number of latent dimensions across models
    latent_dims = 10
    # The number of epochs used for deep learning based models
    epoch_num = 500

    fr_dcca = Wrapper(recon_p=1e-4,
                   cca_p=1,
                   js_p=1,
                   patience=20,
                   batch_size=1024,
                   latent_dims=latent_dims,
                   epoch_num=epoch_num,
                   view1_hidden=[1024, 1024, 1024, 784],
                   view2_hidden=[1024, 1024, 1024, 784])
    fr_dcca.fit()

