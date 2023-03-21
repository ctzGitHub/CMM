import torch, os
from torch import nn
import torch.nn.functional as F
from models import resnet1d_wang as rd_wang


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyNet(nn.Module):
    """
    Combining 12 lead ecg on a network (actually not 50)
    """

    def __init__(self, in_channels = 12, num_classes = 6, encoder_fea_dim = 128,
                 blocks_sizes=[64, 128, 128, 128, 128], verbose=False):
        super().__init__()
        self.taubu_squeeze_linear = nn.Linear(500, encoder_fea_dim)
        self.pooling_1 = nn.AdaptiveAvgPool1d(1)
        self.pooling_2 = nn.AdaptiveAvgPool1d(1)
        self.pooling_3 = nn.AdaptiveAvgPool1d(1)
        self.squeeze_1 = nn.Sequential(
            nn.Linear(encoder_fea_dim*2, encoder_fea_dim//2),
            nn.ReLU(),
            nn.Linear(encoder_fea_dim//2, encoder_fea_dim)

        )
        self.squeeze_2 = nn.Sequential(
            nn.Linear(encoder_fea_dim*2, encoder_fea_dim//2),
            nn.ReLU(),
            nn.Linear(encoder_fea_dim//2, encoder_fea_dim)

        )
        self.squeeze_3 = nn.Sequential(
            nn.Linear(encoder_fea_dim*2, encoder_fea_dim//2),
            nn.ReLU(),
            nn.Linear(encoder_fea_dim//2, encoder_fea_dim)

        )

        self.encoder_1 = rd_wang.resnet1d_wang(num_classes=encoder_fea_dim, input_channels = in_channels)
        self.encoder_2 = rd_wang.resnet1d_wang(num_classes=encoder_fea_dim, input_channels = in_channels)
        self.encoder_3 = rd_wang.resnet1d_wang(num_classes=encoder_fea_dim, input_channels = in_channels)
        # self.decoder = nn.Sequential(
        #     nn.Linear(encoder_fea_dim * 6, encoder_fea_dim*3),
        #     nn.LeakyReLU(),
        #     nn.Linear(encoder_fea_dim*3, num_classes),
        # )
        self.decoder = nn.Linear(encoder_fea_dim*3, num_classes)
        self.verbose = verbose
        self.attention_1 = nn.Sequential(
            nn.Linear(encoder_fea_dim * 2, encoder_fea_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_fea_dim, 1),
        )
        self.attention_2 = nn.Sequential(
            nn.Linear(encoder_fea_dim * 2, encoder_fea_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_fea_dim, 1),
        )
        self.attention_3 = nn.Sequential(
            nn.Linear(encoder_fea_dim * 2, encoder_fea_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_fea_dim, 1),
        )


    def forward(self, inputs1_scale1, inputs1_scale2, inputs1_scale3, inputs2):

        # edition2:
        # tab_logit, tab_feat, M_loss = self.tabnet(inputs2)
        inputs1_scale1 = inputs1_scale1.transpose(1, 2)
        inputs1_scale2 = inputs1_scale2.transpose(1, 2)
        inputs1_scale3 = inputs1_scale3.transpose(1, 2)
        inputs2 = self.taubu_squeeze_linear(inputs2)
        H_1 = [self.encoder_1(inputs1_scale1[:, i, :, :]).view((inputs1_scale1.shape[0], -1, 1)) for i in range(inputs1_scale1.shape[1])]
        H_2 = [self.encoder_2(inputs1_scale2[:, i, :, :]).view((inputs1_scale2.shape[0], -1, 1)) for i in range(inputs1_scale2.shape[1])]
        H_3 = [self.encoder_3(inputs1_scale3[:, i, :, :]).view((inputs1_scale3.shape[0], -1, 1)) for i in range(inputs1_scale3.shape[1])]
        H_1 = torch.cat(H_1, dim=2)
        H_2 = torch.cat(H_2, dim=2)
        H_3 = torch.cat(H_3, dim=2)
        ave_H_1 =self.pooling_1(H_1)
        ave_H_2 = self.pooling_2(H_2)
        ave_H_3 = self.pooling_3(H_3)
        ave_H_1 = torch.transpose(ave_H_1, 1, 2)
        ave_H_2 = torch.transpose(ave_H_2, 1, 2)
        ave_H_3 = torch.transpose(ave_H_3, 1, 2)
        H_1 = torch.transpose(H_1, 1, 2)
        H_2 = torch.transpose(H_2, 1, 2)
        H_3 = torch.transpose(H_3, 1, 2)
        # H.shape = torch.Size([64, 60, 128])
        # ave_H.shape = torch.Size([64, 1, 128])
        # inputs2.shape = torch.Size([64, 128])
        ave_H_1 = ave_H_1.reshape(ave_H_1.shape[0], ave_H_1.shape[2])
        ave_H_2 = ave_H_2.reshape(ave_H_2.shape[0], ave_H_2.shape[2])
        ave_H_3 = ave_H_3.reshape(ave_H_3.shape[0], ave_H_3.shape[2])

        global_feat_1 = torch.cat((ave_H_1, inputs2), 1)
        global_feat_1 = self.squeeze_1(global_feat_1)
        global_feat_1 = global_feat_1.reshape(global_feat_1.shape[0], -1, global_feat_1.shape[1])
        global_feat_1 = global_feat_1.expand_as(H_1)
        feat_1 = torch.cat([H_1, global_feat_1], dim=2)
        global_feat_2 = torch.cat((ave_H_2, inputs2), 1)
        global_feat_2 = self.squeeze_2(global_feat_2)
        global_feat_2 = global_feat_2.reshape(global_feat_2.shape[0], -1, global_feat_2.shape[1])
        global_feat_2 = global_feat_2.expand_as(H_2)
        feat_2 = torch.cat([H_2, global_feat_2], dim=2)
        global_feat_3 = torch.cat((ave_H_3, inputs2), 1)
        global_feat_3 = self.squeeze_3(global_feat_3)
        global_feat_3 = global_feat_3.reshape(global_feat_3.shape[0], -1, global_feat_3.shape[1])
        global_feat_3 = global_feat_3.expand_as(H_3)
        feat_3 = torch.cat([H_3, global_feat_3], dim=2)


        attention_1 = self.attention_1(feat_1)
        attention_1 = F.softmax(attention_1, dim=1)
        A_1 = torch.transpose(attention_1, 1, 2)
        M_1 = torch.bmm(A_1, H_1)
        # M_1 = torch.bmm(A_1, feat_1)
        M_1 = M_1.reshape(M_1.size(0), -1)

        attention_2 = self.attention_2(feat_2)
        attention_2 = F.softmax(attention_2, dim=1)
        A_2 = torch.transpose(attention_2, 1, 2)
        M_2 = torch.bmm(A_2, H_2)
        # M_2 = torch.bmm(A_2, feat_2)
        M_2 = M_2.reshape(M_2.size(0), -1)

        attention_3 = self.attention_3(feat_3)
        attention_3 = F.softmax(attention_3, dim=1)
        A_3 = torch.transpose(attention_3, 1, 2)
        M_3 = torch.bmm(A_3, H_3)
        # M_3 = torch.bmm(A_3, feat_3)
        M_3 = M_3.reshape(M_3.size(0), -1)

        M = torch.cat((M_1, M_2, M_3), 1)
        M = self.decoder(M)





        return M
        # return M, tab_logit, M_loss





if __name__ == '__main__':
    '''
    input = torch.randn(64, 5, 12, 200)
    info = torch.randn(16, 2)
    SE_ECGNet = ECGBagResNet(12, num_classes=9, n_segments=4)
    output = SE_ECGNet(input)
    print(output.shape)
    '''

    inputs1_scale1 = torch.randn(64, 12, 60, 180)
    inputs1_scale2 = torch.randn(64, 12, 60, 90)
    inputs1_scale3 = torch.randn(64, 12, 60, 45)
    inputs2 = torch.randn(64, 500)
    #info = torch.randn(16, 2)
    SE_ECGNet = MyNet()
    SE_ECGNet = SE_ECGNet.to(device)
    inputs1_scale1 = inputs1_scale1.to(device)
    inputs1_scale2 = inputs1_scale2.to(device)
    inputs1_scale3 = inputs1_scale3.to(device)

    inputs2 = inputs2.to(device)
    with torch.no_grad():
        output = SE_ECGNet(inputs1_scale1, inputs1_scale2, inputs1_scale3, inputs2)
    # print(output.shape)
