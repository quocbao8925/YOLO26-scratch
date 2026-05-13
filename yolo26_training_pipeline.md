# YOLO26 Training & Validation Pipeline — From Scratch?

> [!IMPORTANT]
> You do NOT need to rebuild everything. The training/validation pipeline is made of
> **two layers**: a generic engine (reusable as-is or easy to replicate) and
> task-specific pieces (small, well-scoped). Only the **loss function** requires
> serious work.

---

## Source File Map — Training & Validation

```
c:\Users\ASUS\Downloads\MIMIC26\ultralytics\ultralytics\
│
├── engine/
│   ├── trainer.py       ← BaseTrainer  (the main training loop)
│   └── validator.py     ← BaseValidator (the main val loop)
│
├── models/yolo/detect/
│   ├── train.py         ← DetectionTrainer(BaseTrainer)  task-specific overrides
│   └── val.py           ← DetectionValidator(BaseValidator)  task-specific overrides
│
└── utils/
    ├── loss.py          ← E2ELoss, v8DetectionLoss, TaskAlignedAssigner  ⚠️ HARD
    ├── tal.py           ← make_anchors, dist2bbox, TaskAlignedAssigner
    ├── metrics.py       ← DetMetrics, box_iou, ConfusionMatrix
    ├── nms.py           ← non_max_suppression
    ├── torch_utils.py   ← ModelEMA, EarlyStopping, one_cycle, select_device
    └── ops.py           ← xywh2xyxy, scale_boxes, xyxy2xywh
```

---

## Component-by-Component Analysis

### 1. Training Loop — `engine/trainer.py`

**Source:** `engine/trainer.py` — `BaseTrainer` class L67–1069  
**Task override:** `models/yolo/detect/train.py` — `DetectionTrainer` L24–252

The training loop structure (`_do_train`, L376) is:
```
setup → for each epoch:
    scheduler.step()
    model.train()
    for each batch:
        preprocess_batch()      ← train.py L107
        forward + loss          ← model(batch) or model(img) + model.loss()
        scaler.scale(loss).backward()
        optimizer_step()        ← trainer.py L739
        EMA update
    validate()
    save_model()
    EarlyStopping check
```

**What you need to replicate:**

| Piece | Source | Difficulty | Notes |
|-------|--------|------------|-------|
| Training loop skeleton | `engine/trainer.py` L376–584 | ✅ Easy | Standard PyTorch loop |
| `preprocess_batch()` | `models/yolo/detect/train.py` L107 | ✅ Trivial | `/255`, `.float()`, multi-scale resize |
| `optimizer_step()` | `engine/trainer.py` L739 | ✅ Easy | `unscale → clip_grad → step → zero_grad → EMA` |
| AMP scaler | `engine/trainer.py` L340 | ✅ Easy | `torch.amp.GradScaler` |
| Warmup LR | `engine/trainer.py` L425–439 | ✅ Easy | `np.interp` over iterations |
| LR scheduler | `engine/trainer.py` L248–254 | ✅ Easy | `LambdaLR` + linear/cosine |
| Gradient accumulation | `engine/trainer.py` L281, L480 | ✅ Easy | `accumulate = nbs / batch_size` |
| `save_model()` | `engine/trainer.py` L642 | ✅ Easy | `torch.save({epoch, ema, optimizer, ...})` |

---

### 2. `ModelEMA` — Exponential Moving Average

**Source:** `utils/torch_utils.py`

```python
# SOURCE: utils/torch_utils.py  (ModelEMA class)
# Standalone re-implementation — ~20 lines
class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = deepcopy(model).eval()
        self.updates = 0
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()
```

---

### 3. Optimizer — `build_optimizer()`

**Source:** `engine/trainer.py` L800+ (`build_optimizer` method)

The key logic: **param groups split** into (bias, BN weights, other weights) with different weight decays.

```python
# SOURCE: engine/trainer.py  build_optimizer() method
# Simplified standalone version
def build_optimizer(model, lr=0.01, momentum=0.937, decay=5e-4):
    g_bn, g_bias, g_weight = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_bias.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bn.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_weight.append(v.weight)

    optimizer = torch.optim.SGD(g_bn,   lr=lr, momentum=momentum, nesterov=True)
    optimizer.add_param_group({'params': g_weight, 'weight_decay': decay})
    optimizer.add_param_group({'params': g_bias})
    return optimizer
```

> [!TIP]
> The ultralytics code also supports Adam/AdamW/RAdam. For a minimal build, SGD with momentum is sufficient and matches YOLO defaults.

---

### 4. LR Scheduling

**Source:** `utils/torch_utils.py` (`one_cycle`) + `engine/trainer.py` L248–254

```python
# SOURCE: utils/torch_utils.py  one_cycle()
# Cosine annealing (used when cos_lr=True)
def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

# SOURCE: engine/trainer.py  L253  (linear, used by default)
lf = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
```

---

### 5. ⚠️ Loss Function — The Hard Part

**Source:** `utils/loss.py`  
**Entry point for YOLO26:** `E2ELoss` (because `end2end=True` in yaml)

```
E2ELoss (utils/loss.py)
├── one2many branch → v8DetectionLoss
│   ├── TaskAlignedAssigner (utils/tal.py)  ← assigns GT to anchors
│   ├── BboxLoss            (utils/loss.py) ← CIoU + DFL
│   └── varifocal_loss      (utils/loss.py) ← classification
└── one2one branch  → Hungarian matching loss
    └── HungarianMatcher    (utils/loss.py)
```

**Minimal path (skip O2O, use only O2M):**

```python
# SOURCE: utils/loss.py  v8DetectionLoss.__call__()
# The 3 losses computed are:
loss_box  # CIoU bounding box regression loss
loss_cls  # BCE or varifocal classification loss
loss_dfl  # Distribution Focal Loss (skipped if reg_max=1)
```

| Sub-component | Source | Lines | Difficulty |
|---------------|--------|-------|------------|
| `TaskAlignedAssigner` | `utils/tal.py` | ~200 | ⚠️ Medium |
| `BboxLoss` (CIoU) | `utils/loss.py` | ~60 | ✅ Easy |
| `varifocal_loss` | `utils/loss.py` | ~20 | ✅ Easy |
| `HungarianMatcher` (O2O) | `utils/loss.py` | ~100 | ⚠️ Hard |
| `E2ELoss` wrapper | `utils/loss.py` | ~80 | ✅ Easy once above done |

---

### 6. DataLoader

**Source:** `models/yolo/detect/train.py` L79 (`get_dataloader`) + `data/` package

The data package is large and ultralytics-specific (mosaic, augmentations, letterbox).

**Options:**

| Option | Effort | Notes |
|--------|--------|-------|
| Reuse ultralytics `data/` package | Zero | Just `from ultralytics.data import build_dataloader, build_yolo_dataset` |
| Write minimal YOLO dataloader | Medium (~1 day) | Load images + YOLO `.txt` labels, letterbox, basic augment |
| Use Albumentations + custom Dataset | Medium | What your existing pipeline likely already does |

**Minimal custom dataloader skeleton:**
```python
# Replaces: models/yolo/detect/train.py L65-105
class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.imgs   = sorted(Path(img_dir).glob("*.jpg"))
        self.labels = [Path(label_dir) / f"{p.stem}.txt" for p in self.imgs]
        self.img_size = img_size

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.imread(str(self.imgs[i]))[..., ::-1]  # BGR→RGB
        img, ratio, pad = letterbox(img, self.img_size)  # resize+pad
        img = torch.from_numpy(img).permute(2,0,1).float() / 255
        labels = self._load_labels(self.labels[i], ratio, pad)
        return {"img": img, "cls": labels[:,0], "bboxes": labels[:,1:], ...}
```

---

### 7. Validation Loop — `engine/validator.py` + `models/yolo/detect/val.py`

**Source:** `engine/validator.py` (BaseValidator) + `models/yolo/detect/val.py` L21–532

The key steps in `DetectionValidator`:

| Step | Source | What it does |
|------|--------|-------------|
| `preprocess()` | `val.py` L63 | `/255`, `.half()` |
| `model(img)` | forward pass | runs inference |
| `postprocess()` | `val.py` L107 | calls `nms.non_max_suppression()` |
| `update_metrics()` | `val.py` L170 | computes TP/FP per image |
| `_process_batch()` | `val.py` L292 | `box_iou()` matching |
| `get_stats()` | `val.py` L262 | aggregates mAP50, mAP50-95 |

---

### 8. NMS — `utils/nms.py`

**Source:** `utils/nms.py` — `non_max_suppression()`

Replaceable with `torchvision.ops.nms`:

```python
# SOURCE: utils/nms.py  non_max_suppression()
# Standalone replacement using torchvision
from torchvision.ops import nms as torch_nms

def simple_nms(preds, conf_thresh=0.25, iou_thresh=0.45):
    # preds: [B, 4+nc, anchors]  (after Detect.forward inference)
    results = []
    for pred in preds.transpose(-1, -2):      # [anchors, 4+nc]
        boxes  = pred[:, :4]                  # xyxy
        scores = pred[:, 4:].max(-1)          # max class score
        mask   = scores.values > conf_thresh
        boxes, scores_v, cls_i = boxes[mask], scores.values[mask], scores.indices[mask]
        keep = torch_nms(boxes, scores_v, iou_thresh)
        results.append({"bboxes": boxes[keep], "conf": scores_v[keep], "cls": cls_i[keep]})
    return results
```

---

### 9. Metrics — `utils/metrics.py`

**Source:** `utils/metrics.py` — `DetMetrics`, `box_iou`, `ConfusionMatrix`

```python
# SOURCE: utils/metrics.py  box_iou()
# ~20 lines, easy to copy verbatim
def box_iou(box1, box2):
    # box1: [N,4] xyxy,  box2: [M,4] xyxy
    inter = (torch.min(box1[:,None,2:], box2[:,2:]) -
             torch.max(box1[:,None,:2], box2[:,:2])).clamp(0).prod(2)
    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])
    return inter / (area1[:,None] + area2 - inter)
```

`mAP` calculation (`DetMetrics`) is ~300 lines but is **pure NumPy** — no PyTorch, no ultralytics-specific code. You can copy it verbatim.

---

### 10. EarlyStopping

**Source:** `utils/torch_utils.py` — `EarlyStopping` class

```python
# SOURCE: utils/torch_utils.py  EarlyStopping class (~30 lines)
class EarlyStopping:
    def __init__(self, patience=50):
        self.best_fitness = 0.0
        self.best_epoch   = 0
        self.patience     = patience

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.best_epoch   = epoch
        stop = (epoch - self.best_epoch) >= self.patience
        return stop

    @property
    def possible_stop(self):
        return (self.best_epoch + self.patience) == self.current_epoch
```

---

## Summary: What to Build vs Reuse

| Component | Source File | Build from scratch? | Effort |
|-----------|-------------|---------------------|--------|
| Training loop | `engine/trainer.py` L376 | ✅ Easy to write | 2 hrs |
| `preprocess_batch` | `detect/train.py` L107 | ✅ Trivial | 15 min |
| `ModelEMA` | `utils/torch_utils.py` | ✅ 20 lines | 20 min |
| Optimizer split | `engine/trainer.py` L800 | ✅ Easy | 30 min |
| LR schedule | `utils/torch_utils.py` | ✅ 5 lines | 10 min |
| AMP + grad clip | `engine/trainer.py` L739 | ✅ Standard PyTorch | 10 min |
| `save_model` / `load_model` | `engine/trainer.py` L642 | ✅ Easy | 20 min |
| `EarlyStopping` | `utils/torch_utils.py` | ✅ 30 lines | 15 min |
| NMS | `utils/nms.py` | ✅ Use torchvision | 10 min |
| `box_iou` | `utils/metrics.py` | ✅ 20 lines | 10 min |
| `DetMetrics` / mAP | `utils/metrics.py` | ✅ Copy verbatim | 1 hr |
| DataLoader | `data/` package | ⚠️ Medium | 4–8 hrs |
| **`TaskAlignedAssigner`** | **`utils/tal.py`** | ⚠️ Medium | **3–5 hrs** |
| **`v8DetectionLoss`** | **`utils/loss.py`** | ⚠️ Hard | **4–6 hrs** |
| **`HungarianMatcher` (O2O)** | **`utils/loss.py`** | ⚠️ Hard | **3–4 hrs** |

---

## Minimal Standalone Training Loop Template

```python
# Combines all sources above into one standalone script
import torch, math
from pathlib import Path
from copy import deepcopy

# -- Your modules (from the architecture artifact) --
from yolo26_modules import build_yolo26n

model = build_yolo26n(nc=80).cuda()
optimizer = build_optimizer(model, lr=0.01)          # see Section 3
scaler    = torch.amp.GradScaler("cuda")
ema       = ModelEMA(model)                          # see Section 2
stopper   = EarlyStopping(patience=50)               # see Section 10

lrf = 0.01
lf  = lambda x: max(1 - x / epochs, 0) * (1 - lrf) + lrf
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lf)

criterion = E2ELoss(model)   # SOURCE: utils/loss.py

for epoch in range(epochs):
    scheduler.step()
    model.train()

    for i, batch in enumerate(train_loader):
        # preprocess — SOURCE: detect/train.py L107
        imgs = batch["img"].cuda().float() / 255

        # forward + loss — SOURCE: engine/trainer.py L447-451
        with torch.autocast("cuda"):
            preds = model(imgs)
            loss, loss_items = criterion(preds, batch)

        # backward — SOURCE: engine/trainer.py L459
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad()
        ema.update(model)

    # validate every epoch — SOURCE: engine/trainer.py L531-534
    metrics, fitness = validate(ema.ema, val_loader)

    # save — SOURCE: engine/trainer.py L642
    torch.save({"epoch": epoch, "ema": ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(), "fitness": fitness},
               "last.pt")

    if stopper(epoch, fitness): break
```

> [!TIP]
> Since your conversations `3807b542` and `b5bac9d1` show you already have a working training loop, **you only need to integrate `TaskAlignedAssigner` + `v8DetectionLoss`** from `utils/loss.py` and `utils/tal.py` — everything else you already have.
