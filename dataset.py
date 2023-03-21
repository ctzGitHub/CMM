import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from data_process import STpeter, interSTpeter
from config import config




class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, signals_scale1: np.ndarray, signals_scale2: np.ndarray, signals_scale3: np.ndarray, labels: np.ndarray, tabular_features:np.ndarray):
        super(ECGDataset, self).__init__()
        self.data1 = signals_scale1
        self.data2 = signals_scale2
        self.data3 = signals_scale3
        self.data4 = tabular_features
        self.label = labels
        self.num_classes = self.label.shape[1]

        self.cls_num_list = np.sum(self.label, axis=0)

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        x3 = self.data3[index]
        x4 = self.data4[index]
        y = self.label[index]

        #x1 = x1.transpose()
        x1 = torch.tensor(x1.copy(), dtype=torch.float)
        x1 = x1.transpose(0, 1)
        x2 = torch.tensor(x2.copy(), dtype=torch.float)
        x2 = x2.transpose(0, 1)
        x3 = torch.tensor(x3.copy(), dtype=torch.float)
        x3 = x3.transpose(0, 1)

        #x2 = x2.transpose()
        x4 = torch.tensor(x4.copy(), dtype=torch.float)

        y = torch.tensor(y, dtype=torch.float)
        y = y.squeeze()
        return x1, x2, x3, x4, y

    def __len__(self):
        return len(self.data1)





class DownLoadECGData:
    '''
        All experiments data
    '''

    def __init__(self, experiment_name, task, datafolder, sampling_frequency=100, min_samples=0,
                 train_fold=8, val_fold=9, test_fold=10):
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.experiment_name = experiment_name
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency



def load_datasets(datafolder=None, experiment=None):
    '''
    Load the final dataset
    '''
    experiment = experiment

    if datafolder == 'D:/cl/数据集/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/inter/':
        X_train1, X_train2, X_train3, X_train4, y_train, X_test1, X_test2, X_test3, X_test4, y_test, tabular_features_train, tabular_features_test = interSTpeter()


    elif datafolder == 'D:/cl/数据集/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/':
        X_train1, X_train2, X_train3, X_train4, y_train, X_test1, X_test2, X_test3, X_test4, y_test, tabular_features_train, tabular_features_test = STpeter()


    ds_train = ECGDataset(X_train1, X_train2, X_train3, y_train, tabular_features_train)
    ds_test = ECGDataset(X_test1, X_test2, X_test3, y_test, tabular_features_test)

    num_classes = ds_train.num_classes
    train_dataloader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False)

    return train_dataloader, test_dataloader, num_classes


if __name__ == '__main__':
    pass
    train_dataloader, test_dataloader, num_classes = load_datasets(datafolder='D:/cl/数据集/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/')
