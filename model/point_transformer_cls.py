import torch
import torch.nn as nn

from model.point_transformer_blocks import TransitionDown, PointTransformerBlock


class PointTransformerCls(nn.Module):
    def __init__(self, blocks, in_channels=6, num_classes=40):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes, share_planes = in_channels, [32, 64, 128, 256, 512], 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride[0], nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride[1], nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride[2], nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride[3], nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride[4], nsample[4])

        self.cls = nn.Sequential(nn.Linear(planes[4], 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(256, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(128, num_classes))

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = [TransitionDown(self.in_planes, planes, stride, nsample)]
        self.in_planes = planes
        for _ in range(blocks):
            layers.append(PointTransformerBlock(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def forward(self, x):
        xyz = x[..., :3].contiguous()

        xyz1, features1 = self.enc1([xyz, x])
        xyz2, features2 = self.enc2([xyz1, features1])
        xyz3, features3 = self.enc3([xyz2, features2])
        xyz4, features4 = self.enc4([xyz3, features3])
        xyz5, features5 = self.enc5([xyz4, features4])

        res = self.cls(features5.mean(dim=1))
        return res


class PointTransformerCls26(PointTransformerCls):
    def __init__(self, **kwargs):
        super(PointTransformerCls26, self).__init__([1, 1, 1, 1, 1], **kwargs)


class PointTransformerCls38(PointTransformerCls):
    def __init__(self, **kwargs):
        super(PointTransformerCls38, self).__init__([1, 2, 2, 2, 2], **kwargs)


class PointTransformerCls50(PointTransformerCls):
    def __init__(self, **kwargs):
        super(PointTransformerCls50, self).__init__([1, 2, 3, 5, 2], **kwargs)


if __name__ == '__main__':
    model = PointTransformerCls38(in_channels=6, num_classes=40).cuda()
    x = torch.randn(16, 1024, 6).cuda()
    res = model(x)
