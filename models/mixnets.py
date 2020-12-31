import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
from collections import OrderedDict

#Swish Activation
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        #self.sigmoid = nn.Sigmoid()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(x.sigmoid())
        else:
            return x.mul(x.sigmoid())

#SE Block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, fc=False, act_type="swish"):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = fc
        if self.fc:
            self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                    Swish(inplace=True) if act_type == "swish" else nn.ReLU(inplace = True),
                                    nn.Linear(channel // reduction, channel, bias=False),
                                    nn.Sigmoid())
        else:
            self.fc = nn.Sequential(nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, bias=True),
                                    Swish(inplace=True) if act_type == "swish" else nn.ReLU(inplace = True),
                                    nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, bias=True),
                                    nn.Sigmoid())
    def forward(self, x): 
        y = self.avg_pool(x) # BxCx1x1
        if self.fc == False:
            y = self.fc(y).view(x.size(0), x.size(1), 1, 1)
        if self.fc == True:
            y = self.fc(y.view(x.size(0), x.size(1))).view(x.size(0), x.size(1), 1, 1)
        return x * y


#Conv Block
class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 groups=1, dilate=1, act_type="swish"):

        super(ConvBlock, self).__init__()
        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate
        padding = ((kernel_size - 1) // 2) * dilate
        
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels=out_planes, 
                                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                                  dilation=dilate, groups=groups, bias=False),
                                        nn.BatchNorm2d(num_features=out_planes, eps=1e-3, momentum=0.01),
                                        Swish(inplace=True) if act_type == "swish" else nn.ReLU(inplace=True)
                                       )
    def forward(self, x):
        return self.conv_block(x)

#Grouped Point-Wise Convolution
class GPConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_sizes):
        super(GPConv, self).__init__()
        self.num_groups = len(kernel_sizes)
        assert in_planes % self.num_groups == 0
        sub_in_dim = in_planes // self.num_groups
        sub_out_dim = out_planes // self.num_groups

        self.group_point_wise = nn.ModuleList()
        for _ in kernel_sizes:
            self.group_point_wise.append(nn.Conv2d(sub_in_dim, sub_out_dim,
                                                   kernel_size=1, stride=1, padding=0,
                                                   groups=1, dilation=1, bias=False))

    def forward(self, x):
        if self.num_groups == 1:
            return self.group_point_wise[0](x)

        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        mix = [self.group_point_wise[stream](chunks[stream]) for stream in range(self.num_groups)]
        return torch.cat(mix, dim=1)

#Mixed Depth-wise Convolution
class MDConv(nn.Module):
    def __init__(self, in_planes, kernel_sizes, stride=1, dilate=1):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_sizes)
        assert in_planes % self.num_groups == 0
        sub_hidden_dim = in_planes // self.num_groups

        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate

        self.mixed_depth_wise = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = ((kernel_size - 1) // 2) * dilate
            self.mixed_depth_wise.append(nn.Conv2d(sub_hidden_dim, sub_hidden_dim,
                                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                                   groups=sub_hidden_dim, dilation=dilate, bias=False))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depth_wise[0](x)

        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        mix = [self.mixed_depth_wise[stream](chunks[stream]) for stream in range(self.num_groups)]
        return torch.cat(mix, dim=1)


#MixBlock for MixNet
class MixBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expand_ratio,  
                 exp_kernel_sizes, kernel_sizes, poi_kernel_sizes, stride, dilate,
                 reduction_ratio=4, dropout_rate=0.2, act_type="swish"):
        super(MixBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.expand_ratio = expand_ratio

        self.groups = len(kernel_sizes)
        self.use_se = (reduction_ratio is not None) and (reduction_ratio > 1)
        self.use_residual = in_planes == out_planes and stride == 1

        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate
        hidden_dim = in_planes * expand_ratio

        if expand_ratio != 1:
            self.expansion = nn.Sequential(
                    GPConv(in_planes, hidden_dim, kernel_sizes=exp_kernel_sizes),
                    nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
                    Swish(inplace=True) if act_type == "swish" else nn.ReLU(inplace=True)
                    )
        self.depth_wise = nn.Sequential(
                MDConv(hidden_dim, kernel_sizes=kernel_sizes, stride=stride,  dilate=dilate),
                nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
                Swish(inplace=True) if act_type == "swish" else nn.ReLU(inplace=True)
                )

        if self.use_se:
            self.se_block = SELayer(hidden_dim, reduction=reduction_ratio, fc=True, act_type=act_type)

        self.point_wise = nn.Sequential(
                GPConv(hidden_dim, out_planes, kernel_sizes=poi_kernel_sizes),
                nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01)                        
                ) 
    def forward(self, x):
        res = x

        if self.expand_ratio != 1:
            x = self.expansion(x)

        x = self.depth_wise(x)

        
        if self.use_se:
            x = self.se_block(x)

        x = self.point_wise(x)

    
        if self.use_residual:
            if self.training and (self.dropout_rate is not None):
                x = F.dropout2d(input=x, p=self.dropout_rate,
                                training=self.training, inplace=True)
            x = x + res

        return x

#MixNet

class MixNet(nn.Module):
    def __init__(self, arch="s", num_classes=1000):
        super(MixNet, self).__init__()

        params = {
            's': (16, [
                # t, c,  n, k,                ek,     pk,     s, d,  a,      se
                [1, 16,  1, [3],              [1],    [1],    1, 1, "relu",  None],
                [6, 24,  1, [3],              [1, 1], [1, 1], 2, 1, "relu",  None],
                [3, 24,  1, [3],              [1, 1], [1, 1], 1, 1, "relu",  None],
                [6, 40,  1, [3, 5, 7],        [1],    [1],    2, 1, "swish", 12],
                [6, 40,  3, [3, 5],           [1, 1], [1, 1], 1, 1, "swish", 12],
                [6, 80,  1, [3, 5, 7],        [1],    [1, 1], 2, 1, "swish", 24],
                [6, 80,  2, [3, 5],           [1],    [1, 1], 1, 1, "swish", 24],
                [6, 120, 1, [3, 5, 7],        [1, 1], [1, 1], 1, 1, "swish", 12],
                [3, 120, 2, [3, 5, 7, 9],     [1, 1], [1, 1], 1, 1, "swish", 12],
                [6, 200, 1, [3, 5, 7, 9, 11], [1],    [1],    2, 1, "swish", 12],
                [6, 200, 2, [3, 5, 7, 9],     [1],    [1, 1], 1, 1, "swish", 12]
            ], 1.0, 1.0, 0.2),
            'm': (24, [
                # t, c,  n, k,            ek,     pk,     s, d,  a,      se
                [1, 24,  1, [3],          [1],    [1],    1, 1, "relu",  None],
                [6, 32,  1, [3, 5, 7],    [1, 1], [1, 1], 2, 1, "relu",  None],
                [3, 32,  1, [3],          [1, 1], [1, 1], 1, 1, "relu",  None],
                [6, 40,  1, [3, 5, 7, 9], [1],    [1],    2, 1, "swish", 12],
                [6, 40,  3, [3, 5],       [1, 1], [1, 1], 1, 1, "swish", 12],
                [6, 80,  1, [3, 5, 7],    [1],    [1],    2, 1, "swish", 24],
                [6, 80,  3, [3, 5, 7, 9], [1, 1], [1, 1], 1, 1, "swish", 24],
                [6, 120, 1, [3],          [1],    [1],    1, 1, "swish", 12],
                [3, 120, 3, [3, 5, 7, 9], [1, 1], [1, 1], 1, 1, "swish", 12],
                [6, 200, 1, [3, 5, 7, 9], [1],    [1],    2, 1, "swish", 12],
                [6, 200, 3, [3, 5, 7, 9], [1],    [1, 1], 1, 1, "swish", 12]
            ],  1.0, 1.0, 0.25),
            'l': (24, [
                # t, c,  n, k,            ek,     pk,     s, d,  a,      se
                [1, 24,  1, [3],          [1],    [1],    1, 1, "relu",  None],
                [6, 32,  1, [3, 5, 7],    [1, 1], [1, 1], 2, 1, "relu",  None],
                [3, 32,  1, [3],          [1, 1], [1, 1], 1, 1, "relu",  None],
                [6, 40,  1, [3, 5, 7, 9], [1],    [1],    2, 1, "swish", 12],
                [6, 40,  3, [3, 5],       [1, 1], [1, 1], 1, 1, "swish", 12],
                [6, 80,  1, [3, 5, 7],    [1],    [1],    2, 1, "swish", 24],
                [6, 80,  3, [3, 5, 7, 9], [1, 1], [1, 1], 1, 1, "swish", 24],
                [6, 120, 1, [3],          [1],    [1],    1, 1, "swish", 12],
                [3, 120, 3, [3, 5, 7, 9], [1, 1], [1, 1], 1, 1, "swish", 12],
                [6, 200, 1, [3, 5, 7, 9], [1],    [1],    2, 1, "swish", 12],
                [6, 200, 3, [3, 5, 7, 9], [1],    [1, 1], 1, 1, "swish", 12]
            ], 1.3, 1.0, 0.25),
        }
        stem_planes, settings, width_multi, depth_multi, self.dropout_rate = params[arch]
        out_channels = self._round_filters(stem_planes, width_multi)
        self.mod1 = ConvBlock(3, out_channels, kernel_size=3, stride=2,
                              groups=1, dilate=1, act_type="relu")

        in_channels = out_channels
        drop_rate = self.dropout_rate
        
        backbone = []
        mod_id = 0
        for t, c, n, k, ek, pk, s, d, a, se in settings:
            out_channels = self._round_filters(c, width_multi)
            repeats = self._round_repeats(n, depth_multi)

            if self.dropout_rate:
                drop_rate = self.dropout_rate * float(mod_id+1) / len(settings)

            # Create blocks for module
            blocks = []
            for block_id in range(repeats):
                stride = s if block_id == 0 else 1
                dilate = d if stride == 1 else 1

                blocks.append(("block%d" % (block_id + 1), MixBlock(in_channels, out_channels,
                                                                    expand_ratio=t, exp_kernel_sizes=ek,
                                                                    kernel_sizes=k, poi_kernel_sizes=pk,
                                                                    stride=stride, dilate=dilate,
                                                                    reduction_ratio=se,
                                                                    dropout_rate=drop_rate,
                                                                    act_type=a)))

                in_channels = out_channels
            backbone.append(("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks))))
            mod_id += 1

        self.BackBone = nn.Sequential(OrderedDict(backbone))

        self.last_channels = 1536
        self.last_feat = ConvBlock(in_channels, self.last_channels,
                                   kernel_size=1, stride=1,
                                   groups=1, dilate=1, act_type="relu")

        self.classifier = nn.Linear(self.last_channels, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    @staticmethod
    def _make_divisible(value, divisor=8):
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def _round_filters(self, filters, width_multi):
        if width_multi == 1.0:
            return filters
        return int(self._make_divisible(filters * width_multi))

    @staticmethod
    def _round_repeats(repeats, depth_multi):
        if depth_multi == 1.0:
            return repeats
        return int(math.ceil(depth_multi * repeats))

    def forward(self, x):
        x = self.mod1(x)
        x = self.BackBone(x)
        x = self.last_feat(x)

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.last_channels)
        if self.training and (self.dropout_rate is not None):
            x = F.dropout(input=x, p=self.dropout_rate,
                          training=self.training, inplace=True)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    import os
    import time
    from torchstat import stat
    from pytorch_memlab import MemReporter

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    arch = "l"
    img_preparam = {"s": (224, 0.875), "m": (224, 0.875), "l": (224, 0.875)}
    net_h = img_preparam[arch][0]
    model = MixNet(arch=arch, num_classes=1000)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1,
                                momentum=0.90, weight_decay=1.0e-4, nesterov=True)

    # stat(model, (3, net_h, net_h))

    model = model.cuda().train()
    loss_func = nn.CrossEntropyLoss().cuda()
    dummy_in = torch.randn(2, 3, net_h, net_h).cuda().requires_grad_()
    dummy_target = torch.ones(2).cuda().long().cuda()
    reporter = MemReporter(model)

    optimizer.zero_grad()
    dummy_out = model(dummy_in)
    loss = loss_func(dummy_out, dummy_target)
    print('========================================== before backward ===========================================')
    reporter.report()

    loss.backward()
    optimizer.step()
    print('========================================== after backward =============================================')
    reporter.report()
