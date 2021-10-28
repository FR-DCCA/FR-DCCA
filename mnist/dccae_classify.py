import random
import copy
import sklearn.metrics
import torch
from numpy import mean
from torch import optim
from torch.utils.data import TensorDataset
import numpy as np
from models import models
from read_data_validation import MNISTValidation
from sklearn import svm
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class Wrapper:

    def __init__(self,
                 latent_dims: int = 2,
                 learning_rate=1e-3,
                 epoch_num: int = 1,
                 batch_size: int = 16,
                 lam=0,
                 patience: int = 10,
                 view1_hidden: list = None,
                 view2_hidden: list = None):
        self.latent_dims = latent_dims
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.device = torch.device("cuda:0")
        self.patience = patience
        self.batch_size = batch_size
        self.view1_hidden = view1_hidden
        self.view2_hidden = view2_hidden
        self.lam = lam

    def fit(self):
        self.seed_torch(seed=24)
        train = MNISTValidation(set_name='train')
        val = MNISTValidation(set_name='val')
        test = MNISTValidation(set_name='test')

        view1_train = train.train_page_data
        view2_train = train.train_link_data

        # Data Loader for easy mini-batch return in training
        train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=self.batch_size,
                                                   num_workers=5, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=self.batch_size,
                                                  num_workers=5, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=self.batch_size,
                                                 num_workers=5, shuffle=True)

        self.model = models.DCCAE(
            view1_size=view1_train.shape[-1],
            view2_size=view2_train.shape[-1],
            view1_hidden=self.view1_hidden,
            view2_hidden=self.view2_hidden,
            lam=self.lam,
            latent_dims=self.latent_dims,
            device=self.device)

        # svm classify
        self.clf_ = svm.SVC(C=1.0, gamma='auto_deprecated')

        model_params = sum(p.numel() for p in self.model.parameters())
        best_model = copy.deepcopy(self.model.state_dict())
        # print(self.model)
        print("Number of model parameters {}".format(model_params))
        self.model.double().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        min_val_loss = self.latent_dims
        epochs_no_improve = 0
        early_stop = False

        for epoch in range(self.epoch_num):
            if early_stop == False:
                epoch_train_loss, epoch_train_recon_loss, \
                epoch_train_cca_loss = self.train_epoch(train_loader)

                epoch_val_loss, epoch_val_recon_loss, epoch_val_cca_loss, \
                epoch_val_acc, epoch_val_f1 = self.val_epoch(val_loader)
                # current model to generate representation and 50000 for train svm 10000 for test
                epoch_test_acc, epoch_test_f1 = self.test_by_svm(train_dataloader=train_loader, test_dataloader=test_loader)
                print('====> Epoch: {} train loss: {:.4f} val loss: {:.4f} val acc: {:.4f} test acc: {:.4f} test f1: {:.4f}'.format(
                        epoch, epoch_train_loss, epoch_val_loss, epoch_val_acc, epoch_test_acc, epoch_test_f1))

                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    print('Min loss %0.2f' % min_val_loss)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == self.patience:
                        print('Early stopping!')
                        early_stop = True
                        self.model.load_state_dict(best_model)
                        real_epoch = epoch + 1


         # best val model to test num times and get mean acc
        print("-------------------------test--------------------------")
        test_acc = 0.0
        test_f1 = 0.0
        test_num = 5
        for i in range(test_num):
            test_acc_, test_f1_ = self.test_by_svm(train_dataloader=train_loader, test_dataloader=test_loader)
            test_acc += test_acc_
            test_f1 += test_f1_
        print("test 10000 sample one time acc, f1 :", test_acc / test_num, test_f1 / test_num)
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
        recon_loss = 0
        cca_loss = 0
        for batch_idx, (x, y, label) in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            x = x.double()
            y = y.double()
            x, y = x.to(self.device), y.to(self.device)
            model_outputs = self.model(x, y)
            view1 = model_outputs[0]
            loss_recon_cca, loss_recon, loss_cca = self.model.loss(x, y, *model_outputs)
            loss_recon_cca.backward()
            train_loss += loss_recon_cca.item()
            recon_loss += loss_recon.item()
            cca_loss += loss_cca.item()
            self.optimizer.step()
            self.clf_.fit(view1.cpu().data.numpy(), label.cpu().data.numpy())
        return train_loss / len(train_dataloader), \
               recon_loss / len(train_dataloader), \
               cca_loss / len(train_dataloader)

    def val_epoch(self, val_dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_recon_loss = 0
            val_cca_loss = 0
            total_val_acc = 0
            total_val_f1 = 0
            count = 0
            for batch_idx, (x, y, val_label) in enumerate(val_dataloader):
                count += 1
                x, y = x.to(self.device), y.to(self.device)
                x = x.double()
                y = y.double()
                model_outputs = self.model(x, y)
                view1 = model_outputs[0]
                # cal loss
                loss_recon_cca, loss_recon, loss_cca = self.model.loss(x, y, *model_outputs)
                val_loss += loss_recon_cca.item()
                val_recon_loss += loss_recon.item()
                val_cca_loss += loss_cca.item()
                # cal f1 and acc
                pred = self.clf_.predict(view1.cpu().data.numpy())
                val_acc = sklearn.metrics.accuracy_score(val_label.cpu().data.numpy(), pred)
                val_f1 = sklearn.metrics.f1_score(
                    val_label.cpu().data.numpy(), pred, average='macro')
                total_val_acc = total_val_acc + val_acc
                total_val_f1 = total_val_f1 + val_f1

        return val_loss / len(val_dataloader), \
               val_recon_loss / len(val_dataloader), \
               val_cca_loss / len(val_dataloader), \
               total_val_acc / count, \
               total_val_f1 / count

    # use the best model to model the train data and test data,
    # and use svm to train the model result (train data) one time,
    # and then classify all test data one time
    def test_by_svm(self, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        test_view1 = np.empty(shape=(0, self.latent_dims))  # use only view1 model result to classify
        test_label = np.empty(shape=0)
        train_view1 = np.empty(shape=(0, self.latent_dims))  # use only view1 model result to classify
        train_label = np.empty(shape=0)
        # for batch get the all model result -> test data
        for batch_idx, (x, y, label) in enumerate(test_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.double()
            y = y.double()
            model_outputs = self.model(x, y)
            view1 = model_outputs[0]
            view1 = view1.cpu().data.numpy()
            label = label.cpu().data.numpy()
            test_view1 = np.concatenate([test_view1, view1], axis=0)
            test_label = np.concatenate([test_label, label], axis=0)

        # for batch get the all model result -> train data
        for batch_idx, (x, y, label) in enumerate(train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.double()
            y = y.double()
            model_outputs = self.model(x, y)
            view1 = model_outputs[0]
            view1 = view1.cpu().data.numpy()
            label = label.cpu().data.numpy()
            train_view1 = np.concatenate([train_view1, view1], axis=0)
            train_label = np.concatenate([train_label, label], axis=0)

        # svm classify -> train on train data and test on test data
        self.clf_.fit(train_view1, train_label)
        predict = self.clf_.predict(test_view1)
        test_acc = sklearn.metrics.accuracy_score(test_label, predict)
        test_f1 = sklearn.metrics.f1_score(test_label, predict, average='macro')
        return test_acc, test_f1


if __name__ == '__main__':

    # The number of latent dimensions across models
    latent_dims = 10

    # The number of epochs used for deep learning based models
    epoch_num = 500

    dccae = Wrapper(lam=1e-4,
                    patience=20,
                    batch_size=1024,
                    latent_dims=latent_dims,
                    epoch_num=epoch_num,
                    view1_hidden=[1024, 1024, 1024, 784],
                    view2_hidden=[1024, 1024, 1024, 784])
    dccae.fit()

