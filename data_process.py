import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import wfdb
import warnings
warnings.filterwarnings("ignore")
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from sklearn.decomposition import PCA





def data_slice(data, num_sample):
    data_process = []
    for dat in data:
        if dat.shape[0] < num_sample:
            dat = resample(dat, num_sample, axis=0)
        elif dat.shape[0] > num_sample:
            dat = resample(dat, num_sample, axis=0)
        if dat.shape[1] != 12:
            dat = dat[:, 0:12]

        data_process.append(dat)
    return np.array(data_process)

def preprocess_signals(X_train, X_test):
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    return apply_standardizer(X_train, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


#STpeter  instance data
def preprosessSTpeter():
    for folder in ['st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/headatatr/']:
        filenames = os.listdir('D:/cl/数据集/' + folder)
        data = []
        for filename in tqdm(filenames):
            if filename.split('.')[1] == 'dat':
                name = filename.split('.')[0]
                record = wfdb.rdrecord('D:/cl/数据集/' + folder + name)
                ECGdata = record.p_signal
                signal_annotation = wfdb.rdann('D:/cl/数据集/' + folder + name, "atr", sampfrom=0,
                                               sampto=462600)

                for i in range(60 * (len(signal_annotation.sample) // 60 - 1)+1):
                    ventricular_signal = ECGdata[
                                         signal_annotation.sample[i + 1] - 90:signal_annotation.sample[i + 1] + 90]

                    data.append(ventricular_signal)
        data = np.array(data)

    return data


def loaddatas():
    b = preprosessSTpeter()
    data = []
    len = b.shape[0] // 60
    for i in range(2593):
        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            # b[j] = np.array(b[j])
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)

    data = torch.stack(data)
    data1 = data.transpose(2, 3)

    data = []
    for i in range(2594, 2628):

        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)
    data = torch.stack(data)
    data2 = data.transpose(2, 3)

    data = []
    for i in range(2629, len - 1):

        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)

    data = torch.stack(data)
    data3 = data.transpose(2, 3)

    data = torch.cat([data1, data2], dim=0)
    data = torch.cat([data, data3], dim=0)

    return data
#STpeter  instance data


#STpeter bag data & label(multi_hot)
def prosessSTpeter():
    label_dict = {'N': 0, 'V': 1, 'A': 2, 'F': 3, 'Q': 4, 'n': 0, 'R': 5, 'B': 0, 'S': 2}
    ecg_counter = 0
    for folder in ['st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/headatatr/']:
        filenames = os.listdir('D:/cl/数据集/' + folder)
        a = []
        lables = []
        for filename in tqdm(filenames):
            if filename.split('.')[1] == 'dat':
                name = filename.split('.')[0]
                record = wfdb.rdrecord('D:/cl/数据集/' + folder + name)
                ECGdata = record.p_signal
                signal_annotation = wfdb.rdann('D:/cl/数据集/' + folder + name, "atr", sampfrom=0, sampto=462600)
                b = []
                for i in range(len(signal_annotation.sample) // 60 -1):
                    ventricular_signal = ECGdata[signal_annotation.sample[60 * i+1] - 95:signal_annotation.sample[                                                                                                   60 * (i + 1)] + 95]
                    ventricular_signal = ventricular_signal.astype(float)
                    beat_lable = str(signal_annotation.symbol)
                    beat_lable = beat_lable.split("', '")
                    beat_lables = beat_lable[60 * i + 1:60 * (i + 1)]
                    a.append(ventricular_signal)
                    b.append(beat_lables)

                b = np.array(b)
                for i in range(len(b)):
                    ecg_counter += 1
                    d = np.unique(b[i])
                    if len(d) > 1:
                        d = d[~np.isin(d, 'N')]
                        d = d[~np.isin(d, '+')]
                        d = d[~np.isin(d, 'j')]
                    elif len(d) == 1:
                        d = d
                    for i in range(len(d)):
                        d[i] = label_dict[d[i]]
                    d = list(d)
                    d = list(map(int, d))
                    d = np.array(d)
                    lables.append(d)
        datas = np.array(a)
        lables = np.array(lables)
    mlb = MultiLabelBinarizer(classes=[i for i in range(6)])
    y = mlb.fit_transform(lables)
    lable = np.array(y)
    datas = np.delete(datas, 2592)
    lable = np.delete(lable, 2592, 0)
    datas = np.delete(datas, 2626)
    lable = np.delete(lable, 2626, 0)
    return datas, lable


#STPETER inter train instance
def load_STintertrain():
    for folder in ['st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/inter/train/']:
        filenames = os.listdir('D:/cl/数据集/' + folder)
        data = []
        for filename in tqdm(filenames):
            if filename.split('.')[1] == 'dat':
                name = filename.split('.')[0]
                record = wfdb.rdrecord('D:/cl/数据集/' + folder + name)
                ECGdata = record.p_signal
                signal_annotation = wfdb.rdann('D:/cl/数据集/' + folder + name, "atr", sampfrom=0,
                                               sampto=462600)
                for i in range(60 * (len(signal_annotation.sample) // 60 - 1)+1):
                    ventricular_signal = ECGdata[
                                         signal_annotation.sample[i + 1] - 90:signal_annotation.sample[i + 1] + 90]

                    data.append(ventricular_signal)
        data = np.array(data)

    return data


def loadintertrain():
    b = load_STintertrain()
    len = b.shape[0] // 60

    data = []
    for i in range(len):

        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            hj = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(hj)
        datae = torch.stack(datae)
        data.append(datae)
    data = torch.stack(data)
    data = data.transpose(2, 3)

    return data

def load_intertraindatas():
    label_dict = {'N': 0, 'V': 1, 'A': 2, 'F': 3, 'Q': 4, 'n': 0, 'R': 5, 'B': 0, 'S': 2}
    ecg_counter = 0
    for folder in ['st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/inter/train/']:
        filenames = os.listdir('D:/cl/数据集/' + folder)
        a = []
        lables = []
        for filename in tqdm(filenames):
            if filename.split('.')[1] == 'dat':
                name = filename.split('.')[0]
                record = wfdb.rdrecord('D:/cl/数据集/' + folder + name)
                ECGdata = record.p_signal
                signal_annotation = wfdb.rdann('D:/cl/数据集/' + folder + name, "atr", sampfrom=0,
                                               sampto=462600)

                b = []
                for i in range(len(signal_annotation.sample) // 60 - 1):
                    ventricular_signal = ECGdata[signal_annotation.sample[60 * i + 1] - 96:signal_annotation.sample[
                                                                                               60 * (i + 1)] + 96]
                    ventricular_signal = ventricular_signal.astype(float)
                    beat_lable = str(signal_annotation.symbol)
                    beat_lable = beat_lable.split("', '")
                    beat_lables = beat_lable[60 * i + 1:60 * (i + 1)]
                    a.append(ventricular_signal)
                    b.append(beat_lables)

                b = np.array(b)
                for i in range(len(b)):
                    ecg_counter += 1
                    d = np.unique(b[i])
                    if len(d) > 1:
                        d = d[~np.isin(d, 'N')]
                        d = d[~np.isin(d, '+')]
                        d = d[~np.isin(d, 'j')]
                    elif len(d) == 1:
                        d = d
                    for i in range(len(d)):
                        d[i] = label_dict[d[i]]
                    d = list(d)
                    d = list(map(int, d))  # 使用内置map返回一个map对象，再用list将其转换为列表
                    d = np.array(d)
                    lables.append(d)
        datas = np.array(a)
        lables = np.array(lables)
    mlb = MultiLabelBinarizer(classes=[i for i in range(6)])
    y = mlb.fit_transform(lables)
    lable = np.array(y)

    return datas, lable
#STPETER inter train data &;lables


#STPETER inter test instance
def load_STintertest():
    for folder in ['st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/inter/test/']:
        filenames = os.listdir('D:/cl/数据集/' + folder)
        data = []
        for filename in tqdm(filenames):
            if filename.split('.')[1] == 'dat':
                name = filename.split('.')[0]

                record = wfdb.rdrecord('D:/cl/数据集/' + folder + name)
                ECGdata = record.p_signal
                signal_annotation = wfdb.rdann('D:/cl/数据集/' + folder + name, "atr", sampfrom=0,
                                               sampto=462600)

                for i in range(60 * (len(signal_annotation.sample) // 60 - 1)+1):
                    ventricular_signal = ECGdata[
                                         signal_annotation.sample[i + 1] - 90:signal_annotation.sample[i + 1] + 90]

                    data.append(ventricular_signal)
        data = np.array(data)

    return data


def loadintertest():
    b = load_STintertest()
    data = []
    len = b.shape[0] // 60
    for i in range(550):

        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)

    data = torch.stack(data)
    data1 = data.transpose(2, 3)

    data = []
    for i in range(551, 585):

        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)

    data = torch.stack(data)
    data2 = data.transpose(2, 3)

    data = []
    for i in range(586, len):

        datae = []
        for j in range(60 * i + 1, 60 * (i + 1) + 1):
            b[j] = torch.tensor(b[j].copy(), dtype=torch.float)
            datae.append(b[j])
        datae = torch.stack(datae)
        data.append(datae)

    data = torch.stack(data)
    data3 = data.transpose(2, 3)

    data = torch.cat([data1, data2], dim=0)
    data = torch.cat([data, data3], dim=0)

    return data



#STPETER inter test data &;lables
def load_intertestdatas():
    label_dict = {'N': 0, 'V': 1, 'A': 2, 'F': 3, 'Q': 4, 'n': 0, 'R': 5, 'B': 0, 'S': 2}
    ecg_counter = 0
    for folder in ['st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/inter/test/']:
        filenames = os.listdir('D:/cl/数据集/' + folder)
        a = []
        lables = []
        for filename in tqdm(filenames):
            if filename.split('.')[1] == 'dat':
                name = filename.split('.')[0]
                record = wfdb.rdrecord('D:/cl/数据集/' + folder + name)
                ECGdata = record.p_signal
                signal_annotation = wfdb.rdann('D:/cl/数据集/' + folder + name, "atr", sampfrom=0,
                                               sampto=462600)

                b = []
                for i in range(len(signal_annotation.sample) // 60 - 1):
                    ventricular_signal = ECGdata[signal_annotation.sample[60 * i + 1] - 96:signal_annotation.sample[
                                                                                               60 * (i + 1)] + 96]
                    ventricular_signal = ventricular_signal.astype(float)
                    beat_lable = str(signal_annotation.symbol)
                    beat_lable = beat_lable.split("', '")
                    beat_lables = beat_lable[60 * i + 1:60 * (i + 1)]
                    a.append(ventricular_signal)
                    b.append(beat_lables)

                b = np.array(b)
                for i in range(len(b)):
                    ecg_counter += 1
                    d = np.unique(b[i])
                    if len(d) > 1:
                        d = d[~np.isin(d, 'N')]
                        d = d[~np.isin(d, '+')]
                        d = d[~np.isin(d, 'j')]
                    elif len(d) == 1:
                        d = d
                    for i in range(len(d)):
                        d[i] = label_dict[d[i]]
                    d = list(d)
                    d = list(map(int, d))
                    d = np.array(d)
                    lables.append(d)
        datas = np.array(a)
        lables = np.array(lables)
    mlb = MultiLabelBinarizer(classes=[i for i in range(6)])
    y = mlb.fit_transform(lables)
    lable = np.array(y)
    datas = np.delete(datas, 550)
    lable = np.delete(lable, 550, 0)
    datas = np.delete(datas, 584)
    lable = np.delete(lable, 584, 0)

    return datas, lable
#STPETER inter test data &;lables



def multiscale_bag(input, ratio):
    a = []
    for i in range(input.shape[0]):
        b = resample(input[i], int(input[i].shape[0]/ratio), axis=0)
        a.append(b)
    return np.array(a)


def interSTpeter():
    X_train1 = loadintertrain()
    X_train2, X_train3 = multiscale_instances(X_train1)
    X_train4, y_train = load_intertraindatas()
    tabular_features_train = extract_tabular_features(dataloader='interSTpeter_train')

    X_test1 = loadintertest()
    X_test2, X_test3 = multiscale_instances(X_test1)
    X_test4, y_test = load_intertestdatas()
    tabular_features_test = extract_tabular_features(dataloader='interSTpeter_test')

    X_train1, X_test1 = preprocess_signals(X_train1, X_test1)
    X_train2, X_test2 = preprocess_signals(X_train2, X_test2)
    X_train3, X_test3 = preprocess_signals(X_train3, X_test3)
    X_train4, X_test4 = preprocess_signals(X_train4, X_test4)



    return X_train1, X_train2, X_train3, X_train4, y_train, X_test1, X_test2, X_test3, X_test4, y_test, tabular_features_train, tabular_features_test



def extract_tabular_features(dataloader):
    if dataloader == 'STpeter' and os.path.exists('extract_features_EfficientFCParameters.csv'):
        train_features = pd.read_csv("extract_features_EfficientFCParameters.csv")

    elif dataloader == 'interSTpeter_train' and os.path.exists('extract_features_st_inter_train.csv'):
        train_features = pd.read_csv("extract_features_st_inter_train.csv")

    elif dataloader == 'interSTpeter_test' and os.path.exists('extract_features_st_inter_test.csv'):
        train_features = pd.read_csv("extract_features_st_inter_test.csv")
    else:
        if dataloader == 'STpeter':
            instances = loaddatas()
        # instances.shape : torch.Size([2818, 60, 12, 180])
        if dataloader == 'interSTpeter_train':
            instances = loadintertrain()
        if dataloader == 'interSTpeter_test':
            instances = loadintertest()
        a = instances.shape[0]
        instances = instances.reshape(instances.shape[0], -1, instances.shape[2])
        # instances.shape : torch.Size([2818, 10800, 12])
        instances = instances.reshape(-1, instances.shape[2])
        # instances.shape : torch.Size([30434400, 12])
        id = torch.zeros(10800, 1)
        for i in range(1, a):
            temp = torch.ones(10800, 1) * i
            id = torch.cat((id, temp), dim=0)
        temp = torch.arange(0, 60 * 180, 1)
        time = torch.arange(0, 60 * 180, 1)
        time = time.reshape(60 * 180, 1)
        temp = temp.reshape(60 * 180, 1)
        for i in range(a-1):
            time = torch.cat((time, temp), dim=0)
        instances = torch.cat((time, instances), dim=1)
        instances = torch.cat((id, instances), dim=1)
        instances = np.array(instances)
        instances = pd.DataFrame(instances)
        instances.rename(columns={0: "id", 1: "time"}, inplace=True)
        train_features = extract_features(instances, column_id='id', column_sort='time',
                                          default_fc_parameters=EfficientFCParameters())
    if dataloader == 'STpeter' or dataloader == 'MIT' or dataloader == 'interSTpeter_train' or dataloader == 'interSTpeter_test':
        pca = PCA(n_components = 500)
        pca.fit(train_features)
        train_features_reduction = pca.transform(train_features)
        tabular_features = train_features_reduction
    else:
        tabular_features = np.array(train_features)

    return tabular_features
    # return train_features

def multiscale_instances(instances):
    temp1 = []
    temp2 = []
    for i in range(instances.shape[0]):
        for j in range(instances.shape[1]):
            for k in range(instances.shape[2]):
                temp1.append(resample(instances[i][j][k], 90, axis=0))
                temp2.append(resample(instances[i][j][k], 45, axis=0))
#                 采样率分别缩减到4000和2000

    temp1 = np.array(temp1)
    temp2 = np.array(temp2)
    instances_scale2 = temp1.reshape(instances.shape[0], instances.shape[1], instances.shape[2], -1)
    instances_scale3 = temp2.reshape(instances.shape[0], instances.shape[1], instances.shape[2], -1)

    return instances_scale2, instances_scale3


def STpeter():
    datas, labels = prosessSTpeter()
    datas = data_slice(datas, 8000)
    instances_scale1 = loaddatas()
    instances_scale2, instances_scale3 = multiscale_instances(instances_scale1)
    tabular_features = extract_tabular_features(dataloader='STpeter')
    # instances = instances[:, :, 6:12, :]

    data_num = len(labels)
    shuffle_ix = np.random.permutation(np.arange(data_num))
    data = datas[shuffle_ix]
    labels = labels[shuffle_ix]
    instances_scale1 = instances_scale1[shuffle_ix]
    instances_scale2 = instances_scale2[shuffle_ix]
    instances_scale3 = instances_scale3[shuffle_ix]
    tabular_features = tabular_features[shuffle_ix]

    X_train1 = instances_scale1[:int(data_num * 0.7)]
    X_train2 = instances_scale2[:int(data_num * 0.7)]
    X_train3 = instances_scale3[:int(data_num * 0.7)]
    X_train4 = data[:int(data_num * 0.7)]
    y_train = labels[:int(data_num * 0.7)]
    tabular_features_train = tabular_features[:int(data_num * 0.7)]


    X_test1 = instances_scale1[int(data_num * 0.7):]
    X_test2 = instances_scale2[int(data_num * 0.7):]
    X_test3 = instances_scale3[int(data_num * 0.7):]
    X_test4 = data[int(data_num * 0.7):]
    y_test = labels[int(data_num * 0.7):]
    tabular_features_test = tabular_features[int(data_num * 0.7):]

    X_train1, X_test1 = preprocess_signals(X_train1, X_test1)
    X_train2, X_test2 = preprocess_signals(X_train2, X_test2)
    X_train3, X_test3 = preprocess_signals(X_train3, X_test3)
    X_train4, X_test4 = preprocess_signals(X_train4, X_test4)

    return X_train1, X_train2, X_train3, X_train4, y_train, X_test1, X_test2, X_test3, X_test4, y_test, tabular_features_train, tabular_features_test


if __name__ == '__main__':
    pass
    X_train1, X_train2, X_train3, X_train4, y_train, X_test1, X_test2, X_test3, X_test4, y_test, tabular_features_train, tabular_features_test = STpeter()

