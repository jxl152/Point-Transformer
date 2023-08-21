import torch
import torch.nn as nn

from model.point_transformer_blocks import PointTransformerBlock, TransitionDown, TransitionUp, BatchNorm1d_P


class PointTransformerSeg(nn.Module):
    def __init__(self, blocks, in_channels=6, num_classes=13):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes, share_planes = in_channels, [32, 64, 128, 256, 512], 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride[0], nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride[1], nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride[2], nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride[3], nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride[4], nsample[4])

        self.dec5 = self._make_dec(planes[4], 1, share_planes, nsample[4], is_head=True)   # transform p5
        self.dec4 = self._make_dec(planes[3], 1, share_planes, nsample[3]) # fusion p5 and p4
        self.dec3 = self._make_dec(planes[2], 1, share_planes, nsample[2]) # fusion p4 and p3
        self.dec2 = self._make_dec(planes[1], 1, share_planes, nsample[1]) # fusion p3 and p2
        self.dec1 = self._make_dec(planes[0], 1, share_planes, nsample[0]) # fusion p2 and p1

        self.seg = nn.Sequential(nn.Linear(planes[0], planes[0]),
                                 BatchNorm1d_P(planes[0]),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], num_classes))

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = [TransitionDown(self.in_planes, planes, stride, nsample)]
        self.in_planes = planes
        for _ in range(blocks):
            layers.append(PointTransformerBlock(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, planes, blocks, share_planes, nsample, is_head=False):
        layers = [TransitionUp(self.in_planes, None if is_head else planes)]
        self.in_planes = planes
        for _ in range(blocks):
            layers.append(PointTransformerBlock(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def forward(self, px):
        p0 = px[..., :3].contiguous()
        x0 = px

        p1, x1 = self.enc1([p0, x0])
        p2, x2 = self.enc2([p1, x1])
        p3, x3 = self.enc3([p2, x2])
        p4, x4 = self.enc4([p3, x3])
        p5, x5 = self.enc5([p4, x4])

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5])])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4], [p5, x5])])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3], [p4, x4])])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2], [p3, x3])])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1], [p2, x2])])[1]

        res = self.seg(x1)
        return res


class PointTransformerSeg26(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg26, self).__init__([1, 1, 1, 1, 1], **kwargs)


class PointTransformerSeg38(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg38, self).__init__([1, 2, 2, 2, 2], **kwargs)


class PointTransformerSeg50(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg50, self).__init__([1, 2, 3, 5, 2], **kwargs)


if __name__ == '__main__':
    # model = PointTransformerSeg26(in_channels=22, num_classes=50).cuda()
    model = PointTransformerSeg38(in_channels=22, num_classes=50).cuda()
    # model = PointTransformerSeg50(in_channels=22, num_classes=50).cuda()
    x = torch.randn(16, 1024, 22).cuda()
    res = model(x)
