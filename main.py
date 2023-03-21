# -*- coding: utf-8 -*-
import torch, time, os
import utils
from torch import nn, optim

from config import config
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import pandas as pd
from models import CMM
import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = []


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

from dataset import load_datasets

def save_checkpoint(best_map, model, optimizer, epoch):
    print('Model Saving...')
    if config.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_map': best_map,
    }, os.path.join('checkpoints', config.model_name + '_' + config.experiment + '_checkpoint_best.pth'))


def train_epoch(model, optimizer, criterion, train_dataloader, threshold=0.5):
    model.train()
    loss_meter, it_count, f1_meter, acc_meter = 0, 0, 0, 0
    outputs = []
    targets = []
    for inputs1_scale1, inputs1_scale2, inputs1_scale3, inputs2, target in train_dataloader:
        inputs1_scale1 = inputs1_scale1.to(device)
        inputs1_scale2 = inputs1_scale2.to(device)
        inputs1_scale3 = inputs1_scale3.to(device)
        inputs2 = inputs2.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs1_scale1, inputs1_scale2, inputs1_scale3, inputs2)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        output = torch.sigmoid(output)
        acc = utils.cal_accuracy_score(target, output)
        acc_meter += acc


        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())

    f1 = utils.calc_f1(targets, outputs, threshold)
    auc = roc_auc_score(np.array(targets), np.array(outputs), multi_class='ovo')
    recall = utils.calc_recall(targets, outputs, threshold)
    map = utils.compute_mAP(targets, outputs)
    print('train_loss: %.4f, f1: %.4f,  auc: %.4f,  recall: %.4f,  map: %.4f' % (
    loss_meter / it_count, f1, auc, recall, map))
    return loss_meter / it_count, f1, auc, recall, map


def test_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    loss_meter, it_count, f1_meter, acc_meter = 0, 0, 0, 0
    outputs = []
    targets = []
    with torch.no_grad():
        for inputs1_scale1, inputs1_scale2, inputs1_scale3, inputs2, target in val_dataloader:

            inputs1_scale1 = inputs1_scale1.to(device)
            inputs1_scale2 = inputs1_scale2.to(device)
            inputs1_scale3 = inputs1_scale3.to(device)
            inputs2 = inputs2.to(device)
            target = target.to(device)

            output = model(inputs1_scale1, inputs1_scale2, inputs1_scale3, inputs2)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            acc = utils.cal_accuracy_score(target, output)
            acc_meter += acc

            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

        f1 = utils.calc_f1(targets, outputs, threshold)
        auc = roc_auc_score(np.array(targets), np.array(outputs), multi_class='ovo')
        recall = utils.calc_recall(targets, outputs, threshold)
        map = utils.compute_mAP(targets, outputs)
    print('test_loss: %.4f, f1: %.4f,  auc: %.4f,  recall: %.4f,  map: %.4f' % (
    loss_meter / it_count, f1, auc, recall, map))
    return loss_meter / it_count, f1, auc, recall, map


def train(config=config):
    # seed
    setup_seed(config.seed)
    print('torch.cuda.is_available:', torch.cuda.is_available())

    # datasets
    train_dataloader, test_dataloader, num_classes = load_datasets(
        datafolder=config.datafolder,
        experiment=config.experiment,
    )

    # mode
    model = getattr(CMM, config.model_name)()
    print('model_name:{}, num_classes={}'.format(config.model_name, num_classes))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # =========>train<=========
    f1 = []
    auc = []
    recall = []
    map = []
    for epoch in range(1, config.max_epoch + 1):
        print('#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(epoch, config.batch_size,
                                                                             config.lr))

        since = time.time()
        train_loss, train_f1, train_auc, train_recall, train_map = train_epoch(model, optimizer, criterion,
                                                                               train_dataloader, threshold=0.5)

        test_loss, test_f1, test_auc, test_recall, test_map = test_epoch(model, criterion, test_dataloader,
                                                                         threshold=0.5)

        save_checkpoint(test_map, model, optimizer, epoch)

        result_list = [[epoch, train_loss, train_f1, train_auc, train_recall, train_map,
                        test_loss, test_f1, test_auc, test_recall, test_map]]
        loss.append(train_loss)
        f1.append(test_f1)
        auc.append(test_auc)
        recall.append(test_recall)
        map.append(test_map)

        if epoch == 1:
            columns = ['epoch', 'train_loss', 'train_f1', 'train_auc', 'train_recall', 'train_map',
                       'test_loss', 'test_f1', 'test_auc', 'test_recall', 'test_map']

        else:
            columns = ['', '', '', '', '', '', '', '', '', '', '']

        dt = pd.DataFrame(result_list, columns=columns)
        dt.to_csv(config.model_name + config.experiment + 'result.csv', mode='a')

        print('time:%s\n' % (utils.print_time_cost(since)))
        if epoch == 100:
            print('max_f1: %.4f, max_auc: %.4f,  max_recall: %.4f,  max_map: %.4f' % (max(f1), max(auc), max(recall), max(map)))
    max_f1 = max(f1)
    max_auc = max(auc)
    max_recall = max(recall)
    max_map = max(map)
    print('max_f1: %.4f, max_auc: %.4f,  max_recall: %.4f,  max_map: %.4f' % (max_f1, max_auc, max_recall, max_map))


if __name__ == '__main__':
    config.datafolder = 'D:/cl/数据集/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/'
    config.seed = 42
    train(config)


