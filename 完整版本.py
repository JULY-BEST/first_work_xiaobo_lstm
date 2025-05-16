import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.fft
from pytorch_wavelets import DWT1D, IDWT1D
from sa_ode import SelfAttention
import fairseq
import os
from .ACRNN import acrnn
from .AASIST import *
from MyAttention import MyAttention
import librosa
import torchaudio



class ASR_model(nn.Module):
    def __init__(self):
        super(ASR_model, self).__init__()
        cp_path = os.path.join(
            '/home/wangrui/AMSDF-main/pretrained_models/xlsr2_300m.pt')  # Change the pre-trained XLSR model path.
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].cuda()
        self.linear = nn.Linear(1024, 128)
        self.bn = nn.BatchNorm1d(num_features=50)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        emb = self.model(x, mask=False, features_only=True)['x']
        emb = self.linear(emb)
        emb = F.max_pool2d(emb, (4, 2))
        emb = self.bn(emb)
        emb = self.selu(emb)
        return emb  # (bs,frame_number,feat_out_dim)

class SER_model(nn.Module):
    def __init__(self):
        super(SER_model, self).__init__()
        cp_path = os.path.join(
            '/home/wangrui/AMSDF-main/pretrained_models/ser_acrnn.pth')  # Change the pre-trained SER model path.
        model = acrnn().cuda()
        model.load_state_dict(torch.load(cp_path))
        self.bn = nn.BatchNorm1d(num_features=50)
        self.selu = nn.SELU(inplace=True)
        self.model = model

    def forward(self, x):
        emb = self.model(x)
        emb = F.max_pool2d(emb, (3, 4))
        emb = self.bn(emb)
        emb = self.selu(emb)
        return emb

class WaveConv1d(nn.Module):
    def __init__(self):
        super(WaveConv1d, self).__init__()

        # 直接定义参数
        self.in_channels = 128
        self.out_channels = 128
        self.level = 8
        # self.wave = "db4"  # 使用 Daubechies 4
        # self.wave = "haar"  # 使用 Haar 小波
        self.wave = "db6"
        self.mode = "zero"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构造 dummy 数据 (batch_size, channels：特征数, length:帧数)
        dummy_input = torch.randn(12, 128, 50).to(self.device)

        # 定义 DWT1D 和 IDWT1D
        self.dwt1d = DWT1D(wave=self.wave, J=self.level, mode=self.mode).to(self.device)
        self.idwt1d = IDWT1D(wave=self.wave, mode=self.mode).to(self.device)

        # 获取 mode_data 和 coe_data 的维度信息
        mode_data, _ = self.dwt1d(dummy_input)
        self.modes1 = mode_data.shape[-1]

        # 定义 Self-Attention 层
        self.sa_c = SelfAttention(dim=self.in_channels, heads=1)

        # 初始化权重
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1))

    def mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft, x_coeff = self.dwt1d(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1]).to(self.device)
        out_ft = self.mul1d(x_ft, self.weights1)

        x_coeff[-1] = x_coeff[-1].permute(0, 2, 1)
        x_coeff[-1] = self.sa_c(x_coeff[-1])
        x_coeff[-1] = F.gelu(x_coeff[-1])
        x_coeff[-1] = x_coeff[-1].permute(0, 2, 1)
        x_coeff[-1] = self.mul1d(x_coeff[-1], self.weights2)

        x = self.idwt1d((out_ft, x_coeff))
        x = x[:, :, :50]  # 裁剪到 67 帧
        return x

class Block(nn.Module):
    def __init__(self, dim=128):
        super(Block, self).__init__()

        # 使用内嵌参数实例化 WaveConv1d
        self.filter = WaveConv1d()

        # 定义卷积层
        self.conv = nn.Conv1d(dim, dim, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.filter(x)
        x2 = self.conv(x)
        x = x1 + x2
        x = F.gelu(x)
        return x

'''目前效果最好的 小波-lstm'''
class Module(nn.Module):
    def __init__(self, embed_dim=128, depth=4):
        super(Module, self).__init__()
        # 设置随机种子
        self.setup_seed(0)
        """multi-view feature extractor"""
        self.text_view_extract=ASR_model()
        self.emo_view_extract=SER_model()
        # Embedding adjustment layers
        self.embedding_adjustment_audio = nn.Linear(embed_dim, 256)
        self.embedding_adjustment_emo = nn.Linear(embed_dim, 256)

        # LSTM layer
        # self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=True,bidirectional=True)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=True)

        # Attention layer
        self.my_attention = MyAttention(embed_dim=256, num_heads=4)
        self.dense_1 = nn.Linear(256, 128)
        self.batch_normalization_1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.dense_2 = nn.Linear(128, 64)
        self.batch_normalization_2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.dense_3 = nn.Linear(64, 2)

        # Define multiple Block layers
        self.blocks = nn.ModuleList([Block(dim=embed_dim) for _ in range(depth)])

        # Fully connected layers for final classification
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, inputs,inputs2, Freq_aug):
        x=inputs
        x2=inputs2
        """multi-view features"""
        emo_view=self.emo_view_extract(x2) # 10 50 64  (bs,frame_number,feat_out_dim)
        # print(emo_view.shape)
        text_view=self.text_view_extract(x) # 10 50 64
        # print(text_view.shape)
        concat_view = torch.cat([text_view,emo_view], dim=-1) # 10 50 128
        concat_view = concat_view.transpose(1, 2)

        for blk in self.blocks:
            x = blk(concat_view)

        x = x.permute(0, 2, 1) #[1, 67, 128] 每个时间步有128个特征
        # Adjust embedding dimension for LSTM
        x = self.embedding_adjustment_audio(x)
        # Process through LSTM
        x, _ = self.lstm(x)  # [batch_size, frame_number, hidden_size]
        # Apply attention mechanism
        x = x.transpose(0, 1)  # [frame_number, batch_size, hidden_size]
        x = self.my_attention(x)  # [frame_number, batch_size, hidden_size]
        x = x.transpose(0, 1)  # [batch_size, frame_number, hidden_size]
        # Select the last time step
        x = x[:, -1, :]  # [batch_size, hidden_size]

        x = self.dense_1(x)
        x = self.batch_normalization_1(x)
        x = self.dropout1(x)
        x = self.dense_2(x)
        x = self.batch_normalization_2(x)
        x = self.dropout2(x)
        x = self.dense_3(x)
        return x
