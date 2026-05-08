import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive Conv -> BatchNorm -> ReLU layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SODNet(nn.Module):
    def __init__(self, in_channels=3, base=32, dropout=0.3, depth=4):
        super().__init__()
        assert depth in (4, 5), "depth must be 4 or 5"
        self.depth = depth

        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for level in range(depth):
            ch_out = base * (2 ** level)
            self.encoders.append(ConvBlock(ch_in, ch_out))
            ch_in = ch_out
        self.pool = nn.MaxPool2d(2)

        bottleneck_ch = base * (2 ** depth)
        self.bottleneck = ConvBlock(ch_in, bottleneck_ch)
        self.dropout = nn.Dropout2d(p=dropout)

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        ch_prev = bottleneck_ch
        for level in reversed(range(depth)):
            skip_ch = base * (2 ** level)
            self.ups.append(nn.ConvTranspose2d(ch_prev, skip_ch, kernel_size=2, stride=2))
            self.decs.append(ConvBlock(skip_ch + skip_ch, skip_ch))
            ch_prev = skip_ch

        self.out_conv = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x, return_logits=False):
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)
        h = self.dropout(h)

        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))

        logits = self.out_conv(h)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


if __name__ == "__main__":
    for depth in (4, 5):
        model = SODNet(depth=depth)
        dummy = torch.randn(2, 3, 128, 128)
        out = model(dummy)
        print(f"depth={depth}: out={tuple(out.shape)}  "
              f"params={sum(p.numel() for p in model.parameters()):,}")