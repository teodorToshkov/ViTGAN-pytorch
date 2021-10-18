from torch import nn

# CNN Generator

class CNNGenerator(nn.Module):
    def __init__(self, hidden_size, latent_dim):
        super(CNNGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.w = nn.Linear(latent_dim, hidden_size * 2 * 4 * 4, bias=False)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( hidden_size, hidden_size // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( hidden_size // 2, hidden_size // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( hidden_size // 4, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        input = self.w(input).view((-1, self.hidden_size * 2, 4, 4))
        return self.main(input)
