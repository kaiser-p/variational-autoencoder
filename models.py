from torch import nn


class ConvolutionAwareVariationalAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SimpleConvolutionalAE(nn.Module):
    def __init__(self, img_dim: int, bottleneck_dim: int, verbose: bool = False):
        super().__init__()
        self.img_dim = img_dim  # D
        self.bottleneck_dim = bottleneck_dim  # N
        self.verbose = verbose

        self.relu = nn.ReLU()

        self.encoder_layer1_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # Output: [B, C=16, D, D]
        self.encoder_layer1_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [B, C=16, D/2, D/2]

        self.encoder_layer2_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # Output: [B, C=32, D/2, D/2]
        self.encoder_layer2_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [B, C=32, D/4, D/4]

        self.encoder_layer3_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Output: [B, C=64, D/4, D/4]
        self.encoder_layer3_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [B, C=64, D/8, D/8]

        self.encoder_layer4_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Output: [B, C=128, D/8, D/8]
        self.encoder_layer4_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [B, C=128, D/16, D/16]

        self.encoder_layer5_conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # Output: [B, C=128, D/16, D/16]
        self.encoder_layer5_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [B, C=128, D/32, D/32]

        hidden_dim = 128*(img_dim//32)*(img_dim//32)
        self.encoder_layer6_dense = nn.Linear(in_features=hidden_dim, out_features=hidden_dim//4)  # Output: [B, 64 * D/32 * D/32]
        self.encoder_layer7_dense = nn.Linear(in_features=hidden_dim//4, out_features=hidden_dim//16)  # Output: [B, 32 * D/32 * D/32]
        self.encoder_layer8_dense = nn.Linear(in_features=hidden_dim//16, out_features=self.bottleneck_dim)  # Output: [B, N]

        self.latent_space_manipulation = ConvolutionAwareVariationalAE()

        self.decoder_layer8_dense = nn.Linear(in_features=self.bottleneck_dim, out_features=hidden_dim//16)  # Output: [B, 32 * D/32 * D/32]
        self.decoder_layer7_dense = nn.Linear(in_features=hidden_dim//16, out_features=hidden_dim//4)  # Output: [B, 64 * D/32 * D/32]
        self.decoder_layer6_dense = nn.Linear(in_features=hidden_dim//4, out_features=hidden_dim)  # Output: [B, 128 * D/32 * D/32]

        self.decoder_layer5_conv = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)  # Output: [B, C=128, D/16, D/16]
        self.decoder_layer4_conv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)  # Output: [B, C=64, D/8, D/8]
        self.decoder_layer3_conv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)  # Output: [B, C=32, D/4, D/4]
        self.decoder_layer2_conv = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)  # Output: [B, C=16, D/2, D/2]
        self.decoder_layer1_conv = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2, stride=2, padding=0)  # Output: [B, C=3, D, D]

    def forward(self, x):
        if self.verbose:
            print("x:", x.shape)
        x = self.relu(self.encoder_layer1_conv(x))
        if self.verbose:
            print("encoder_layer1_conv:", x.shape)
        x = self.encoder_layer1_pool(x)
        if self.verbose:
            print("encoder_layer1_pool:", x.shape)

        x = self.relu(self.encoder_layer2_conv(x))
        if self.verbose:
            print("encoder_layer2_conv:", x.shape)
        x = self.encoder_layer2_pool(x)
        if self.verbose:
            print("encoder_layer2_pool:", x.shape)

        x = self.relu(self.encoder_layer3_conv(x))
        if self.verbose:
            print("encoder_layer3_conv:", x.shape)
        x = self.encoder_layer3_pool(x)
        if self.verbose:
            print("encoder_layer3_pool:", x.shape)

        x = self.relu(self.encoder_layer4_conv(x))
        if self.verbose:
            print("encoder_layer4_conv:", x.shape)
        x = self.encoder_layer4_pool(x)
        if self.verbose:
            print("encoder_layer4_pool:", x.shape)

        x = self.relu(self.encoder_layer5_conv(x))
        if self.verbose:
            print("encoder_layer5_conv:", x.shape)
        x = self.encoder_layer5_pool(x)
        if self.verbose:
            print("encoder_layer5_pool:", x.shape)

        B, C, D_w, D_h = x.shape
        x = x.view(B, -1)  # [B, C*D_w*D_h]
        if self.verbose:
            print("x.view:", x.shape)

        x = self.relu(self.encoder_layer6_dense(x))
        if self.verbose:
            print("encoder_layer6_dense:", x.shape)
        x = self.relu(self.encoder_layer7_dense(x))
        if self.verbose:
            print("encoder_layer7_dense:", x.shape)
        x = self.relu(self.encoder_layer8_dense(x))
        if self.verbose:
            print("encoder_layer8_dense:", x.shape)

        x = self.latent_space_manipulation(x)
        if self.verbose:
            print("latent_space_manipulation:", x.shape)

        x = self.relu(self.decoder_layer8_dense(x))
        if self.verbose:
            print("decoder_layer8_dense:", x.shape)
        x = self.relu(self.decoder_layer7_dense(x))
        if self.verbose:
            print("decoder_layer7_dense:", x.shape)
        x = self.relu(self.decoder_layer6_dense(x))
        if self.verbose:
            print("decoder_layer6_dense:", x.shape)

        x = x.view(B, C, D_w, D_h)  # [B, C, D_w, D_h]
        if self.verbose:
            print("x.view:", x.shape)

        x = self.relu(self.decoder_layer5_conv(x))
        if self.verbose:
            print("decoder_layer5_conv:", x.shape)
        x = self.relu(self.decoder_layer4_conv(x))
        if self.verbose:
            print("decoder_layer4_conv:", x.shape)
        x = self.relu(self.decoder_layer3_conv(x))
        if self.verbose:
            print("decoder_layer3_conv:", x.shape)
        x = self.relu(self.decoder_layer2_conv(x))
        if self.verbose:
            print("decoder_layer2_conv:", x.shape)
        x = self.relu(self.decoder_layer1_conv(x))
        if self.verbose:
            print("decoder_layer1_conv:", x.shape)
        return x
