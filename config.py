# -*- coding: utf-8 -*-
'''
@time: 2022/4/26
@author: cl
'''


class Config():
    seed = 42

    datafolder = 'D:/cl/数据集/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/inter/'

    #
    '''
    experiment = exp0, exp1, exp1.1, exp1.1.1, exp2, exp3
    '''
    experiment = 'exp_CPSC'


    model_name = 'MyNet'



    batch_size = 64
    n_segments1 = 59
    n_segments2 = 8

    max_epoch = 100

    lr = 0.001

    device_num = 1

config = Config()
