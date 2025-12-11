import torch
import torch.nn as nn
from tinyvit_hw_ops import hw_conv2d_nchw

torch.manual_seed(0)

conv = nn.Conv2d(
    8, 16, kernel_size=3,
    stride=2, padding=1,
    bias=True, dilation=1
)

hw = hw_conv2d_nchw.from_conv(conv)

x = torch.randn(1, 8, 32, 32)

y_ref = conv(x)
y_hw  = hw(x)

print("max diff:", (y_ref - y_hw).abs().max())
print("mean diff:", (y_ref - y_hw).abs().mean())
