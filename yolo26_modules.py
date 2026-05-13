"""YOLO26 neural network modules — ported from ultralytics (torch-only, no ultralytics imports).
Every class/function has a SOURCE comment pointing to the original file and line range.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility helpers
# SOURCE: nn/modules/conv.py  L30-36
# ---------------------------------------------------------------------------
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# SOURCE: utils/ops.py  L144-156
def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/conv.py  L39-89
# ---------------------------------------------------------------------------
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# SOURCE: nn/modules/conv.py  L185-199
class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# SOURCE: nn/modules/conv.py  L616-641
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/block.py  L58-80
# ---------------------------------------------------------------------------
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/block.py  L457-481
# ---------------------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/block.py  L288-319
# ---------------------------------------------------------------------------
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/block.py  L322-345
# ---------------------------------------------------------------------------
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# SOURCE: nn/modules/block.py  L1109-1127
class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/block.py  L1271-1328
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


# SOURCE: nn/modules/block.py  L1331-1378
class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/block.py  L1069-1106
# ---------------------------------------------------------------------------
class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/block.py  L208-237
# ---------------------------------------------------------------------------
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/block.py  L1436-1488
# ---------------------------------------------------------------------------
class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


# ---------------------------------------------------------------------------
# SOURCE: nn/modules/head.py  L26-252  (Detect head)
# SOURCE: utils/tal.py  L400-413     (make_anchors)
# SOURCE: utils/tal.py  L416-425     (dist2bbox)
# ---------------------------------------------------------------------------
def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):
        stride = strides[i]
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


class Detect(nn.Module):
    dynamic = False
    export = False
    format = None
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        return getattr(self, "_end2end", True) and hasattr(self, "one2one_cv2")

    @end2end.setter
    def end2end(self, value):
        self._end2end = value

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x):
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        return dist2bbox(bboxes, anchors, xywh=xywh and not self.end2end, dim=1)

    def postprocess(self, preds):
        boxes, scores = preds.split([4, self.nc], dim=-1)
        index = scores.amax(-1).topk(min(self.max_det, scores.shape[1]))[1].unsqueeze(-1)
        boxes = boxes.gather(1, index.expand(-1, -1, 4))
        scores = scores.gather(1, index.expand(-1, -1, self.nc))
        scores, cls = scores.flatten(1).topk(min(self.max_det, scores.shape[1]))
        return torch.cat([boxes.gather(1, (cls // self.nc).unsqueeze(-1).expand(-1, -1, 4)),
                          scores.unsqueeze(-1), (cls % self.nc).unsqueeze(-1).float()], -1)

    def bias_init(self):
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def set_head_attr(self, max_det=300):
        self.max_det = max_det


# ---------------------------------------------------------------------------
# YOLO26 Model — hardcoded from yolo26.yaml
# Scales: n=[0.50, 0.25, 1024], s=[0.50, 0.50, 1024],
#          m=[0.50, 1.00, 512],  l=[1.00, 1.00, 512],  x=[1.00, 1.50, 512]
# ---------------------------------------------------------------------------
class YOLO26(nn.Module):
    """Complete YOLO26 model hardcoded from yolo26.yaml.
    SOURCE: cfg/models/26/yolo26.yaml + nn/tasks.py parse_model()
    """
    SCALES = {
        "n": (0.50, 0.25, 1024),
        "s": (0.50, 0.50, 1024),
        "m": (0.50, 1.00, 512),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.50, 512),
    }

    def __init__(self, nc=80, scale="n", reg_max=1):
        super().__init__()
        depth, width, max_ch = self.SCALES[scale]
        self.nc = nc
        self.reg_max = reg_max
        self.names = {i: str(i) for i in range(nc)}

        def ch(c):
            return make_divisible(min(c, max_ch) * width, 8)

        def rep(n):
            return max(round(n * depth), 1) if n > 1 else n

        # ---- Backbone (layers 0-10) ----
        self.b0 = Conv(3, ch(64), 3, 2)          # P1/2
        self.b1 = Conv(ch(64), ch(128), 3, 2)    # P2/4
        self.b2 = C3k2(ch(128), ch(256), n=rep(2), c3k=False, e=0.25)
        self.b3 = Conv(ch(256), ch(256), 3, 2)   # P3/8
        self.b4 = C3k2(ch(256), ch(512), n=rep(2), c3k=False, e=0.25)
        self.b5 = Conv(ch(512), ch(512), 3, 2)   # P4/16
        self.b6 = C3k2(ch(512), ch(512), n=rep(2), c3k=True)
        self.b7 = Conv(ch(512), ch(1024), 3, 2)  # P5/32
        self.b8 = C3k2(ch(1024), ch(1024), n=rep(2), c3k=True)
        self.b9 = SPPF(ch(1024), ch(1024), k=5, n=3, shortcut=True)
        self.b10 = C2PSA(ch(1024), ch(1024), n=rep(2))

        # ---- Head (layers 11-22) ----
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.h13 = C3k2(ch(1024) + ch(512), ch(512), n=rep(2), c3k=True)  # cat b10+b6
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.h16 = C3k2(ch(512) + ch(512), ch(256), n=rep(2), c3k=True)   # cat h13+b4 → P3
        self.h17 = Conv(ch(256), ch(256), 3, 2)
        self.h19 = C3k2(ch(256) + ch(512), ch(512), n=rep(2), c3k=True)   # cat h17+h13 → P4
        self.h20 = Conv(ch(512), ch(512), 3, 2)
        self.h22 = C3k2(ch(512) + ch(1024), ch(1024), n=rep(1), c3k=True, e=0.5, attn=True)  # cat h20+b10 → P5

        # ---- Detect ----
        det_ch = (ch(256), ch(512), ch(1024))
        self.detect = Detect(nc=nc, reg_max=reg_max, end2end=True, ch=det_ch)

        # Compute strides via dummy forward
        self._init_strides()
        self.detect.bias_init()

        # Attributes expected by loss
        self.args = None  # set by trainer
        self.model = nn.ModuleList([self.detect])  # for compatibility with loss: model.model[-1]
        self.stride = self.detect.stride

    def _init_strides(self):
        s = 640
        dummy = torch.zeros(1, 3, s, s)
        with torch.no_grad():
            feats = self._forward_backbone_head(dummy)
        self.detect.stride = torch.tensor([s / f.shape[-1] for f in feats])

    def _forward_backbone_head(self, x):
        # Backbone
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        b6 = self.b6(b5)
        b7 = self.b7(b6)
        b8 = self.b8(b7)
        b9 = self.b9(b8)
        b10 = self.b10(b9)
        # Head
        x = self.up1(b10)
        x = torch.cat([x, b6], 1)
        h13 = self.h13(x)
        x = self.up2(h13)
        x = torch.cat([x, b4], 1)
        h16 = self.h16(x)                    # P3
        x = self.h17(h16)
        x = torch.cat([x, h13], 1)
        h19 = self.h19(x)                    # P4
        x = self.h20(h19)
        x = torch.cat([x, b10], 1)
        h22 = self.h22(x)                    # P5
        return [h16, h19, h22]

    def forward(self, x):
        feats = self._forward_backbone_head(x)
        return self.detect(feats)

    def loss(self, batch, preds):
        """Called by the training loop: model.loss(batch, preds) -> (loss, loss_items)."""
        return self.criterion(preds, batch)


def build_yolo26(nc=80, scale="n", reg_max=1):
    model = YOLO26(nc=nc, scale=scale, reg_max=reg_max)
    return model
