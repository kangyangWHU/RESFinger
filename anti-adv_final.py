import sys
sys.path.append("..")

import config as flags
from utils.defense_utils import *
from utils.loader_utils import Cifar10Loader
from utils.attack_utils import CWLoss, to_onehot, to_infhot
import torch
import numpy as np
from watermark.net.wide_resnet import cifar_wide_resnet
from watermark.metric_utils import load_cifar_model, load_tinyimagenet_model, load_imagenet_model
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch import nn as nn
from matplotlib import pyplot as plt
from utils.test_utils import to_plt_data
import random
import os
from kornia.losses import ssim_loss

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AntiAdv():
    def __init__(self, pth, dataset="cifar10", lr=0.001, n_query=100, n_limits=1000,
                 lambd=0.5, epsilon=32/255, batch_size=50, loss_flag="small", correct_flag=False):
        super(AntiAdv, self).__init__()
        self.pth = pth
        self.dataset = dataset
        self.lr = lr
        self.n_query = n_query
        self.n_limits = n_limits
        self.lambd = lambd
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.loss_flag = loss_flag
        self.correct_flag = correct_flag

        self.init()

    def init(self):

        if self.dataset == "cifar10":
            self.mean = flags.cifar10_mean
            self.std = flags.cifar10_std
            self.load_model = load_cifar_model

            _trans = T.Compose([
                                    T.ToPILImage(),
                                    T.ToTensor()
                                ])

            self.loader = Cifar10Loader(200, train_transforms=_trans, num_workers=0, shuffle=True).train_loader

        elif self.dataset == "imagenet":
            self.mean = flags.torch_mean
            self.std = flags.torch_std
            self.load_model = load_imagenet_model

            trans = [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor()
            ]
            self.loader = DataLoader(ImageFolder("F:/datasets/ImageNet12/train5w",  transform= T.Compose(trans)),
                                     batch_size=100, shuffle=True, num_workers=4)
        else:
            self.mean = flags.tinyimagenet_mean
            self.std = flags.tinyimagenet_std
            self.load_model = load_tinyimagenet_model
            self.loader = DataLoader(ImageFolder("F:/datasets/tiny-imagenet/train",  transform=T.ToTensor()),
                                     batch_size=100, shuffle=True, num_workers=0)

        self.pos_model = self.init_model(self.pth)

        self.neg_model_list = self.load_model_form_dir("weights/{}/train_data/negative_models".format(self.dataset))
        self.pos_model_list = self.load_model_form_dir("weights/{}/train_data/positive_models".format(self.dataset))

    def init_model(self, path):
        model = self.load_model(path)
        model = InputTransformModel(model, normalize=(self.mean, self.std))
        model = model.cuda()
        model.eval()
        return model

    def load_model_form_dir(self, path_dir):

        model_list = []
        for file_name in os.listdir(path_dir):
            model = self.init_model(os.path.join(path_dir, file_name))
            model_list.append(model)
        return model_list

    def cw_logit(self, logits, label):

        num_classes = logits.shape[1]
        onehot_label = to_onehot(label, num_classes)
        infhot_label = to_infhot(label, num_classes)

        target_logits = logits[onehot_label.bool()]
        other_max_logits = (logits - infhot_label).max(dim=1)[0]

        return target_logits - other_max_logits

    def get_logits(self, model_list, img, label):

        sum_logits = 0
        for model in model_list:
            # sum_logits += model(img)[0, label]
            # TODO change label logit - other max logit
            logits = model(img)
            sum_logits += self.cw_logit(logits, label)
        return sum_logits/len(model_list)

    def select_samples(self, loss_flag="small", correct=False):
        # collect enough initial samples
        init_data, init_labels = [], []

        all_data = []
        for data, labels in self.loader:
            data, labels = data.cuda(), labels.cuda()

            with torch.no_grad():
                logits = self.pos_model(data)

            for i in range(len(labels)):
                loss = torch.nn.CrossEntropyLoss()(logits[i].unsqueeze(0),
                                            torch.tensor([labels[i]], dtype=torch.long, device=logits.device))

                if (correct and logits[i].argmax() == labels[i]) or (not correct and logits[i].argmax() != labels[i]):
                    all_data.append((loss.item(), data[i].cpu(), labels[i].cpu()))

            #
            if self.dataset == "imagenet" and len(all_data) > 10000:
                break

        reverse_flag = False if loss_flag == "small" else True
        all_data = sorted(all_data, key=lambda x: x[0], reverse=reverse_flag)

        for loss, img, label in all_data:
            init_data.append(img.unsqueeze(0))
            init_labels.append(label)

        init_data = torch.vstack(init_data)
        init_labels = torch.hstack(init_labels)
        return init_data, init_labels


    def gen_queryset(self):

        trans_list = [GaussianBlur(p=0.5), MedianBlur(ksize=3, p=0.5),RandShear(shearx=(0, 0.1), sheary=(0, 0.1), p=0.5),
                      AverageBlur(3, p=0.5), GaussainNoise(0, 0.1, p=0.5)]

        if self.dataset == "cifar10":
            trans_list += [RandTranslate(tx=(0, 5), ty=(0, 5), p=0.5)]
        else:
            trans_list += [RandTranslate(tx=(0, 10), ty=(0, 10), p=0.5)]

        init_data, init_labels = self.select_samples(loss_flag=self.loss_flag, correct=self.correct_flag)
        dataloader = DataLoader(TensorDataset(init_data, init_labels), batch_size=self.batch_size, shuffle=False)
        ## generate pert
        query_set = []
        query_labels = []
        original_set = []
        for imgs, labels in dataloader:
            imgs, labels = imgs.cuda(), labels.cuda()
            # img_show = to_plt_data(imgs[0].data)
            # plt.imshow(img_show)
            # plt.show()

            imgs.requires_grad = True
            optimizer = torch.optim.Adam([imgs], lr=self.lr)
            # optimizer = torch.optim.SGD([img], lr=self.lr, momentum=0.9, nesterov=True)
            original_imgs = torch.clone(imgs.data)
            for iter in range(self.n_limits):

                trans_index = np.random.randint(0, len(trans_list))
                trans_imgs = trans_list[trans_index](imgs)

                pos_logits = self.get_logits(self.pos_model_list, trans_imgs, labels)
                neg_logits = self.get_logits(self.neg_model_list, trans_imgs, labels)
                loss = -(pos_logits - self.lambd*neg_logits)
                loss = loss.mean()

                # loss += beta*ssim_loss(original_img.data, trans_imgs.data, window_size=11)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                imgs.data.clamp_(0, 1)

                # clamp by epsilon
                imgs.data = original_imgs + torch.clamp(
                    imgs.data - original_imgs, -self.epsilon, self.epsilon)

            # img_show = to_plt_data(imgs[0].data)
            # plt.imshow(img_show)
            # plt.show()

            mask = None

            # positive models output the same label as true label
            for pos_model in self.pos_model_list:
                pred_label = pos_model(imgs).argmax(1)
                if mask is None:
                    mask = (pred_label == labels)
                else:
                    mask &= (pred_label == labels)
                # print(pred_label, end="\t")

            # negative models output different labels as true label
            for neg_model in self.neg_model_list:
                pred_label = neg_model(imgs).argmax(1)
                mask &= (pred_label != labels)

                # print(pred_label, end="\t")
            # print("pos label ", end="\t")

            # print("original label:{}".format(labels))
            # print()

            for img, oimg, label in zip(imgs[mask], original_imgs[mask], labels[mask]):
                query_set.append(img.detach().unsqueeze(0).cpu().numpy())
                original_set.append(oimg.detach().unsqueeze(0).cpu().numpy())
                query_labels.append(label)


                if len(query_set) == self.n_query:
                    query_set = np.vstack(query_set)
                    query_set = torch.tensor(query_set, dtype=torch.float32)

                    original_set = np.vstack(original_set)
                    original_set = torch.tensor(original_set, dtype=torch.float32)

                    query_labels = torch.tensor(query_labels, dtype=torch.long)
                    torch.save(query_set, "query_data/{}_antiadv_queryset.pth".format(self.dataset))
                    torch.save(original_set, "query_data/{}_antiadv_original_set.pth".format(self.dataset))
                    torch.save(query_labels, "query_data/{}_antiadv_querylabels.pth".format(self.dataset))
                    exit()

if __name__=="__main__":
    # 设置随机数种子
    setup_seed(20)
    # dataset = "cifar10"
    dataset = "tinyimagenet"  # 1.3  2 not good 1.5 slightly better
    # dataset = "imagenet"
    # for dataset in ["cifar10", "tinyimagenet"]:
    pth = "weights/{}/source_model.pth".format(dataset)
    antiadv = AntiAdv(pth, dataset=dataset, lr=0.01, n_query=100,
                      n_limits=500, lambd=1.0, epsilon=8/255, batch_size=50,
                      loss_flag="small", correct_flag=False)
    antiadv.gen_queryset()