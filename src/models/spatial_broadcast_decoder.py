from torch import nn

from src.lib import deconv_norm_act, SoftPositionEmbed

class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, input_size, resolution, channels, ks, norm, strides, out_channels, add_pos_emb=True):
        super().__init__()

        self.input_size = input_size
        self.resolution = resolution
        self.channels = channels
        self.ks = ks
        self.norm = norm
        self.strides = strides
        self.out_channels = out_channels
        self.add_pos_emb = add_pos_emb

        # Build Decoder
        # Spatial broadcast --> PosEnc --> DeConv CNN
        modules = []
        for i in range(len(self.channels) - 1):
            modules.append(
                deconv_norm_act(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=self.ks,
                    stride=strides[i],
                    norm=self.norm,
                    act='relu'))

        modules.append(nn.Conv2d(
                self.channels[-1], self.out_channels, kernel_size=1, stride=1, padding=0))

        self.decoder = nn.Sequential(*modules)
        if self.add_pos_emb:
            self.decoder_pos_embedding = SoftPositionEmbed(self.input_size,
                                                           self.resolution)
        else:
            self.decoder_pos_embedding = None

    def forward(self, x):
        assert(len(x.shape) == 2)
        out = x.view(x.shape[0], x.shape[1], 1, 1)
        out = out.repeat(1, 1, self.resolution[0], self.resolution[1])

        if self.decoder_pos_embedding is not None:
            out = self.decoder_pos_embedding(out)

        out = self.decoder(out)

        return out
