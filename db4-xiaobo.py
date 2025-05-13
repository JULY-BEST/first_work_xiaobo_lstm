class WaveConv1d(nn.Module):
    def __init__(self):
        super(WaveConv1d, self).__init__()

        # 直接定义参数
        self.in_channels = 128
        self.out_channels = 128
        self.level = 8
        self.wave = "db4"
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
