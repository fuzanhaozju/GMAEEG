import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np
import random
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MyDevice = torch.device('cuda:0')


def init_edge_weight():
    adj_localglobal = loadmat('./adj_mat.mat')
    adj_localglobal = adj_localglobal['adj_mat']

    adj_predefined = np.ones(shape=(adj_localglobal.shape[0], adj_localglobal.shape[1]))

    index = np.argwhere(adj_localglobal == 0)
    adj_predefined[index[:, 0], index[:, 1]] = 0.5
    adj_predefined[range(adj_predefined.shape[0]), range(adj_predefined.shape[0])] = 0

    adj_t = torch.tensor(adj_predefined)
    edge_index = np.array(adj_t.nonzero().t().contiguous())

    initialized_edge_weight = adj_predefined[edge_index[0, :], edge_index[1, :]].reshape(-1, 1)

    return initialized_edge_weight



class adj_update(nn.Module):
    def __init__(self, inc, reduction_ratio):
        super(adj_update, self).__init__()
        self.fc = nn.Sequential(nn.Linear(inc, inc // reduction_ratio, bias = False),
                                nn.ELU(inplace = False),
                                nn.Linear(inc // reduction_ratio, inc, bias = False),
                                nn.Tanh(),
                                nn.ReLU(inplace = False))

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.fc(x)
        x = x.transpose(0, 1)
        return x



class Encoder(torch.nn.Module):
    def __init__(self, num_features, channels=21):
        super(Encoder, self).__init__()

        self.head = nn.Identity()
        self.conv1 = ChebConv(num_features, 256, K=5)
        self.conv2 = ChebConv(256, 64, K=5)

        num_edges = channels * channels - channels
        self.edge_weight = nn.Parameter(torch.tensor(init_edge_weight(), dtype = torch.float32, requires_grad = True))
        self.adj_update = adj_update(num_edges, reduction_ratio=4)

    def forward(self, x, edge_index):
        edge_weight = self.edge_weight
        train_edge_weight = self.adj_update(edge_weight)
        _edge_weight = train_edge_weight
        for i in range(edge_index.shape[-1] // train_edge_weight.shape[0] - 1):
            train_edge_weight = torch.cat((train_edge_weight, _edge_weight), dim=0)

        x = F.relu(self.conv1(x, edge_index, train_edge_weight))
        x = self.conv2(x, edge_index, train_edge_weight)
        x = self.head(x)

        return x, _edge_weight, train_edge_weight



class Encoder_to_Decoder(torch.nn.Module):
    def __init__(self):
        super(Encoder_to_Decoder, self).__init__()
        self.encoder_to_decoder = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        x = self.encoder_to_decoder(x)
        return x



class Decoder(torch.nn.Module):
    def __init__(self, num_features):
        super(Decoder, self).__init__()

        self.head = nn.Identity()
        self.conv1 = ChebConv(64, 256, K=5)
        self.conv2 = ChebConv(256, num_features, K=5)

    def forward(self, x, edge_index, train_edge_weight):
        x = F.relu(self.conv1(x, edge_index, train_edge_weight))
        x = self.conv2(x, edge_index, train_edge_weight)
        x = self.head(x)
        return x



class GMAEEG(nn.Module):
    """
    Graph Masked Autoencoder for EEG
    """

    def __init__(self, masked_num, head):
        super(GMAEEG, self).__init__()
        self.masked_num = masked_num
        self.head = head

        self.encoder = Encoder(num_features=1152)
        self.encoder_to_decoder = Encoder_to_Decoder()
        self.decoder = Decoder(num_features=1152)
        self.smallconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 4), stride=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 4), stride=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 4), stride=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 4), stride=(1, 3)),
        )
        self.midconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 8), stride=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 8), stride=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 8), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 8), stride=(1, 2)),
        )
        self.largeconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 16), stride=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 16), stride=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 16), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 16), stride=(1, 1)),
        )

        self.smalldeconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(1, 4), stride=(1, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 4), stride=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 4), stride=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 4), stride=(1, 3), padding=(0, 1)),
        )
        self.middeconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(1, 8), stride=(1, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 8), stride=(1, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 8), stride=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 8), stride=(1, 3)),
        )
        self.largedeconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(1, 16), stride=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 16), stride=(1, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 16), stride=(1, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 16), stride=(1, 3), padding=(0, 1)),
        )


        # mask and remask token
        self.enc_token = nn.Parameter(torch.FloatTensor(np.random.normal(0, 0.01, size=(500))))
        self.dec_token = nn.Parameter(torch.FloatTensor(np.random.normal(0, 0.01, size=(64))))

        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 2)


    def forward(self, indata):
        # FOR loops instead of .reshape() are used,
        # to ensure operations are conducted on those dimensions with specific physical meanings
        x, edge_index, batch = indata['x'], indata['edge_index'], indata['batch_info']

        if self.head == 'pretrain':
            mask_node_list = random.sample(range(0, 21), self.masked_num)

            masked_x = x.clone()
            noise_x = x.clone()
            small_x = x.clone()
            mid_x = x.clone()
            large_x = x.clone()
            for index_batch in range(0, int((masked_x.shape[0])/21)):
                for index_mask in range(self.masked_num):
                    dice_general = random.sample(range(0, 10), 1)
                    dice_noise = random.sample(range(0, 21), 1)
                    if dice_general[0] < 8:
                        masked_x[mask_node_list[index_mask] + 21 * index_batch, :] = self.enc_token
                    elif dice_general[0] == 8:
                        masked_x[mask_node_list[index_mask] + 21 * index_batch, :] = noise_x[dice_noise[0] + 21 * index_batch, :]
                    # dice_general[0] == 9: 'Unchanged'

            temp_x = torch.FloatTensor(np.zeros(shape=(int((x.shape[0])/21), 1, 21, 500))).to(MyDevice)
            for index_batch in range(0, int((x.shape[0]) / 21)):
                temp_x[index_batch, 0, :, :] = masked_x[index_batch*21:(index_batch+1)*21, :]

            smallconv_out = self.smallconv(temp_x)
            midconv_out = self.midconv(temp_x)
            largeconv_out = self.largeconv(temp_x)

            cat_conv_out = torch.cat((smallconv_out, midconv_out, largeconv_out), dim=3)
            per_conv_out = cat_conv_out.permute(0, 2, 1, 3)
            enc_in = torch.FloatTensor(np.zeros(shape=(int(21*(x.shape[0])/21), cat_conv_out.shape[1]*cat_conv_out.shape[3]))).to(MyDevice)

            for index_batch in range(0, int((x.shape[0]) / 21)):
                for index_channel in range(0, int(cat_conv_out.shape[3])):
                    enc_in[index_batch*21:(index_batch+1)*21, index_channel*cat_conv_out.shape[1]:(index_channel+1)*cat_conv_out.shape[1]] = per_conv_out[index_batch, :, :, index_channel]

            out, _edge_weight, train_edge_weight = self.encoder(enc_in, edge_index)
            out = self.encoder_to_decoder(out)

            for index_batch in range(0, int((masked_x.shape[0])/21)):
                for index_mask in range(self.masked_num):
                    out[mask_node_list[index_mask]+21*index_batch, :] = self.dec_token

            dec_out = self.decoder(out, edge_index, train_edge_weight)

            deconv_in = per_conv_out
            for index_batch in range(0, int((x.shape[0]) / 21)):
                for index_channel in range(0, int(cat_conv_out.shape[3])):
                    deconv_in[index_batch, :, :, index_channel] = dec_out[index_batch * 21:(index_batch + 1) * 21,index_channel * cat_conv_out.shape[1]:(index_channel + 1) * cat_conv_out.shape[1]]
            per_deconv_in = deconv_in.permute(0, 2, 1, 3)

            small_deconv_in = per_deconv_in[:, :, :, 0:smallconv_out.shape[3]]
            mid_deconv_in = per_deconv_in[:, :, :, smallconv_out.shape[3]:smallconv_out.shape[3]+midconv_out.shape[3]]
            large_deconv_in = per_deconv_in[:, :, :, smallconv_out.shape[3]+midconv_out.shape[3]:per_deconv_in.shape[3]]

            smalldeconv_out = self.smalldeconv(small_deconv_in)
            middeconv_out = self.middeconv(mid_deconv_in)
            largedeconv_out = self.largedeconv(large_deconv_in)

            smallreshaped_out = small_x
            midreshaped_out = mid_x
            largereshaped_out = large_x
            for index_batch in range(0, int((x.shape[0]) / 21)):
                smallreshaped_out[index_batch * 21:(index_batch + 1) * 21, :] = smalldeconv_out[index_batch, 0, :, :]
                midreshaped_out[index_batch * 21:(index_batch + 1) * 21, :] = middeconv_out[index_batch, 0, :, :]
                largereshaped_out[index_batch * 21:(index_batch + 1) * 21, :] = largedeconv_out[index_batch, 0, :, :]


            smallreshaped_out = smallreshaped_out.unsqueeze(2)
            midreshaped_out = midreshaped_out.unsqueeze(2)
            largereshaped_out = largereshaped_out.unsqueeze(2)
            cat_reshaped_out = torch.cat((smallreshaped_out, midreshaped_out, largereshaped_out), dim=2)
            reshaped_out = torch.mean(cat_reshaped_out, dim=2)

            return reshaped_out, _edge_weight


        elif self.head == 'finetune':
            origin_x = x.clone()

            temp_x = torch.FloatTensor(np.zeros(shape=(int((x.shape[0]) / 21), 1, 21, 500))).to(MyDevice)
            for index_batch in range(0, int((x.shape[0]) / 21)):
                temp_x[index_batch, 0, :, :] = origin_x[index_batch * 21:(index_batch + 1) * 21, :]

            smallconv_out = self.smallconv(temp_x)
            midconv_out = self.midconv(temp_x)
            largeconv_out = self.largeconv(temp_x)

            cat_conv_out = torch.cat((smallconv_out, midconv_out, largeconv_out), dim=3)
            per_conv_out = cat_conv_out.permute(0, 2, 1, 3)
            enc_in = torch.FloatTensor(np.zeros(shape=(int(21 * (x.shape[0]) / 21), cat_conv_out.shape[1] * cat_conv_out.shape[3]))).to(MyDevice)

            for index_batch in range(0, int((x.shape[0]) / 21)):
                for index_channel in range(0, int(cat_conv_out.shape[3])):
                    enc_in[index_batch * 21:(index_batch + 1) * 21,
                    index_channel * cat_conv_out.shape[1]:(index_channel + 1) * cat_conv_out.shape[1]] = per_conv_out[index_batch, :, :, index_channel]

            out, _edge_weight, train_edge_weight = self.encoder(enc_in, edge_index)

            out = gap(out, batch)
            out = self.fc1(out)
            out = self.fc2(out)

            return out, _edge_weight


        else:
            raise Exception('Invalid head')


if __name__ == "__main__":
    sample_input = {}
    # 672 = batch size (32) * EEG channels (21)         500 = slice time (2 seconds) * sample rate (250 Hz)
    sample_input['x'] = Variable(torch.ones([672, 500])).cuda()
    # 13440 = batch size (32) * number of non-zero elements in adj matrix (21 * 20)
    sample_input['edge_index'] = torch.ones([2, 13440], dtype=torch.int64).cuda()
    # 672 = batch size (32) * EEG channels (21)
    sample_input['batch_info'] = torch.arange(0, 32).repeat_interleave(21).cuda()

    model = GMAEEG(masked_num=11, head='pretrain').cuda()      # head = 'pretrain' / 'finetune'

    output, updated_edge_weight = model(sample_input)

    print('Input shape is:', sample_input['x'].shape)
    print('Output shape is:', output.shape)
    # to get an updated edge weight matrix in 21*21,
    # reshape the out_edge_weight to 21*20 and insert zeros to the diagonal line
    print(updated_edge_weight.shape)
