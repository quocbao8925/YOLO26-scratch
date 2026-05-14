"""YOLO26 standalone training script for Google Colab.
All modules sourced from ultralytics repo — zero ultralytics imports needed.

Usage on Colab:
    !python train26.py --data /content/dataset --nc 1 --epochs 100 --batch 16 --scale n
Dataset structure expected:
    dataset/
      images/train/   images/val/
      labels/train/   labels/val/
Labels: YOLO format  (cls cx cy w h) normalized per line.
"""
import argparse, csv, gc, math, os, random, time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Local imports (same directory)
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo26_modules import build_yolo26, make_anchors, dist2bbox
from yolo26_loss import E2ELoss, xywh2xyxy, bbox_iou


# ============================= Dataset =====================================
# SOURCE: data/base.py letterbox + data/dataset.py YOLODataset (simplified)
# ===========================================================================
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """Resize and pad image to new_shape keeping aspect ratio."""
    shape = im.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, imgsz=640, augment=False):
        self.imgsz = imgsz
        self.augment = augment
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.img_files = sorted([p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts])
        self.label_dir = Path(label_dir)
        assert len(self.img_files) > 0, f"No images found in {img_dir}"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        im = cv2.imread(str(img_path))
        assert im is not None, f"Failed to read {img_path}"
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h0, w0 = im.shape[:2]

        label_path = self.label_dir / f"{img_path.stem}.txt"
        labels = np.zeros((0, 5), dtype=np.float32)
        if label_path.exists():
            raw = np.loadtxt(str(label_path), ndmin=2, dtype=np.float32)
            if raw.size:
                labels = raw[:, :5]  # cls, cx, cy, w, h (normalized)

        if self.augment:
            im, labels = self._augment(im, labels)

        im, ratio, (dw, dh) = letterbox(im, self.imgsz)
        im = im.transpose(2, 0, 1).astype(np.float32) / 255.0
        im = np.ascontiguousarray(im)

        nl = len(labels)
        cls = labels[:, 0:1] if nl else np.zeros((0, 1), dtype=np.float32)
        bboxes = labels[:, 1:5] if nl else np.zeros((0, 4), dtype=np.float32)

        return {
            "img": torch.from_numpy(im),
            "cls": torch.from_numpy(cls),
            "bboxes": torch.from_numpy(bboxes),
            "im_file": str(img_path),
            "ori_shape": (h0, w0),
        }

    def _augment(self, im, labels):
        # Horizontal flip
        if random.random() < 0.5:
            im = np.fliplr(im).copy()
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]
        # HSV augmentation
        if random.random() < 0.5:
            h_gain, s_gain, v_gain = 0.015, 0.7, 0.4
            r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
            hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return im, labels


def collate_fn(batch):
    imgs = torch.stack([b["img"] for b in batch])
    cls_list, bbox_list, bidx_list = [], [], []
    for i, b in enumerate(batch):
        cls_list.append(b["cls"])
        bbox_list.append(b["bboxes"])
        bidx_list.append(torch.full((len(b["cls"]),), i, dtype=torch.float32))
    return {
        "img": imgs,
        "cls": torch.cat(cls_list) if cls_list else torch.zeros((0, 1)),
        "bboxes": torch.cat(bbox_list) if bbox_list else torch.zeros((0, 4)),
        "batch_idx": torch.cat(bidx_list) if bidx_list else torch.zeros(0),
        "im_file": [b["im_file"] for b in batch],
    }


# ======================== Training Utilities ===============================
# SOURCE: utils/torch_utils.py  ModelEMA, EarlyStopping, one_cycle
# SOURCE: engine/trainer.py  build_optimizer, _setup_scheduler
# ===========================================================================
class ModelEMA:
    """SOURCE: utils/torch_utils.py"""
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = deepcopy(model).eval()
        self.updates = 0
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay_fn(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()


class EarlyStopping:
    """SOURCE: utils/torch_utils.py"""
    def __init__(self, patience=50):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.best_epoch = epoch
        return (epoch - self.best_epoch) >= self.patience


def build_optimizer(model, lr=0.01, momentum=0.937, decay=5e-4):
    """SOURCE: engine/trainer.py build_optimizer method"""
    g_bn, g_bias, g_weight = [], [], []
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            g_bias.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bn.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            g_weight.append(v.weight)
    optimizer = torch.optim.SGD(g_bn, lr=lr, momentum=momentum, nesterov=True)
    optimizer.add_param_group({"params": g_weight, "weight_decay": decay})
    optimizer.add_param_group({"params": g_bias})
    return optimizer


# ========================= Simple NMS ======================================
def simple_nms(boxes, scores, iou_threshold=0.45):
    """torchvision-free NMS. boxes: [N,4] xyxy, scores: [N]."""
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = boxes[rest, 0].clamp(min=boxes[i, 0].item())
        yy1 = boxes[rest, 1].clamp(min=boxes[i, 1].item())
        xx2 = boxes[rest, 2].clamp(max=boxes[i, 2].item())
        yy2 = boxes[rest, 3].clamp(max=boxes[i, 3].item())
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a1 + a2 - inter + 1e-7)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ======================== Validation =======================================
def box_iou_matrix(box1, box2, eps=1e-7):
    """box1 [N,4] xyxy, box2 [M,4] xyxy -> [N,M]"""
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


@torch.no_grad()
def validate(model, dataloader, device, nc, criterion=None, conf_thresh=0.25, iou_thresh=0.45):
    """Run validation, return mAP50 and optionally validation loss."""
    model.eval()
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    vloss = None
    nb = len(dataloader)

    for i, batch in enumerate(dataloader):
        imgs = batch["img"].to(device)
        y, preds = model(imgs)  # inference returns (y, preds)

        if criterion:
            _, loss_items = criterion(preds, batch)
            vloss = loss_items if vloss is None else (vloss * i + loss_items) / (i + 1)
            if i % 20 == 0 or i == nb - 1:
                print(f"  [Val {i+1}/{nb}]  vbox={vloss[0]:.4f}  vcls={vloss[1]:.4f}  vdfl={vloss[2]:.4f}")

        # y from end2end postprocess: [bs, max_det, 6] (x1,y1,x2,y2,score,cls)
        for si in range(imgs.shape[0]):
            pred = y[si]
            pred = pred[pred[:, 4] > conf_thresh]

            # Ground truth for this image
            idx = batch["batch_idx"] == si
            tcls = batch["cls"][idx].squeeze(-1).to(device)
            tbox = batch["bboxes"][idx].to(device)
            nl = len(tcls)

            if len(pred) == 0:
                if nl:
                    stats["tp"].append(torch.zeros(0, len(iouv), dtype=torch.bool, device=device))
                    stats["conf"].append(torch.zeros(0, device=device))
                    stats["pred_cls"].append(torch.zeros(0, device=device))
                    stats["target_cls"].append(tcls)
                continue

            # Scale gt boxes to pixel coords
            if nl:
                imgsz = imgs.shape[2]
                tbox_pixel = xywh2xyxy(tbox) * imgsz
            else:
                tbox_pixel = torch.zeros((0, 4), device=device)

            predn_box = pred[:, :4]
            predn_cls = pred[:, 5]
            predn_conf = pred[:, 4]

            stats["conf"].append(predn_conf)
            stats["pred_cls"].append(predn_cls)
            stats["target_cls"].append(tcls)

            if nl == 0:
                stats["tp"].append(torch.zeros(len(pred), len(iouv), dtype=torch.bool, device=device))
                continue

            correct = torch.zeros(len(pred), len(iouv), dtype=torch.bool, device=device)
            iou = box_iou_matrix(tbox_pixel, predn_box)
            correct_class = tcls[:, None] == predn_cls[None, :]
            for j, thr in enumerate(iouv):
                matches = (iou >= thr) & correct_class
                if matches.any():
                    x = torch.nonzero(matches, as_tuple=False)
                    if x.shape[0]:
                        m = torch.cat((x, iou[x[:, 0], x[:, 1]][:, None]), 1)
                        m = m[m[:, 2].argsort(descending=True)]
                        # unique gt
                        _, idx_u = torch.unique(m[:, 0], return_inverse=True)
                        mask = torch.zeros(len(m), dtype=torch.bool, device=device)
                        for fi in range(int(idx_u.max()) + 1):
                            mask[(idx_u == fi).nonzero(as_tuple=True)[0][0]] = True
                        m = m[mask]
                        # unique pred
                        _, idx_u2 = torch.unique(m[:, 1], return_inverse=True)
                        mask2 = torch.zeros(len(m), dtype=torch.bool, device=device)
                        for fi in range(int(idx_u2.max()) + 1):
                            mask2[(idx_u2 == fi).nonzero(as_tuple=True)[0][0]] = True
                        m = m[mask2]
                        correct[m[:, 1].long(), j] = True
            stats["tp"].append(correct)

    if not stats["tp"]:
        return 0.0, vloss

    tp = torch.cat(stats["tp"]).cpu().numpy()
    conf = torch.cat(stats["conf"]).cpu().numpy()
    pred_cls = torch.cat(stats["pred_cls"]).cpu().numpy()
    target_cls = torch.cat(stats["target_cls"]).cpu().numpy()

    # Compute AP per class at IoU=0.5
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)
    ap50 = []
    for c in unique_classes:
        n_gt = (target_cls == c).sum()
        n_pred = (pred_cls == c).sum()
        if n_pred == 0 or n_gt == 0:
            ap50.append(0.0)
            continue
        fpc = (1 - tp[pred_cls == c, 0]).cumsum(0)
        tpc = tp[pred_cls == c, 0].cumsum(0)
        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc + 1e-16)
        # AP via 101-point interpolation (manual trapezoid to avoid numpy deprecation)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        y_interp = np.interp(x, mrec, mpre)
        ap50.append(np.sum((x[1:] - x[:-1]) * (y_interp[1:] + y_interp[:-1]) / 2.0))

    mAP50 = np.mean(ap50) if ap50 else 0.0
    return mAP50, vloss


# ========================= Plot Results ====================================
def plot_results(csv_path, save_dir):
    """Plot training metrics from results.csv and save as results.png."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend for Colab/server
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot. Install with: pip install matplotlib")
        return

    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))

    if not data or 'epoch' not in data:
        print("No data to plot.")
        return

    epochs = data['epoch']
    metrics = [
        ('box_loss', 'Box Loss', 'tab:red'),
        ('cls_loss', 'Cls Loss', 'tab:blue'),
        ('dfl_loss', 'DFL Loss', 'tab:orange'),
        ('mAP50', 'mAP@0.5', 'tab:green'),
        ('lr', 'Learning Rate', 'tab:purple'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('YOLO26 Training Results', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, (key, title, color) in enumerate(metrics):
        if key in data:
            axes[i].plot(epochs, data[key], color=color, linewidth=2)
            axes[i].set_title(title, fontsize=13)
            axes[i].set_xlabel('Epoch')
            axes[i].grid(True, alpha=0.3)
            if 'loss' in key:
                axes[i].set_ylabel('Loss')

    # Combined loss plot
    if all(k in data for k in ('box_loss', 'cls_loss', 'dfl_loss')):
        total = [b + c + d for b, c, d in zip(data['box_loss'], data['cls_loss'], data['dfl_loss'])]
        axes[5].plot(epochs, total, color='black', linewidth=2)
        axes[5].set_title('Total Loss', fontsize=13)
        axes[5].set_xlabel('Epoch')
        axes[5].set_ylabel('Loss')
        axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(save_dir) / 'results.png'
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Results plot saved to {plot_path}")


# ============================ Training =====================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    # Model
    model = build_yolo26(nc=args.nc, scale=args.scale, reg_max=args.reg_max)
    hyp = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5,
        lr0=args.lr, lrf=0.01, momentum=0.937, weight_decay=5e-4,
        warmup_epochs=3.0, warmup_bias_lr=0.1, warmup_momentum=0.8,
        epochs=args.epochs, cos_lr=True,
    )
    model.args = hyp
    model.to(device)

    # Loss
    criterion = E2ELoss(model)
    model.criterion = criterion

    # Datasets
    data_root = Path(args.data)
    train_ds = YOLODataset(
        data_root / "images" / "train", data_root / "labels" / "train",
        imgsz=args.imgsz, augment=True,
    )
    val_ds = YOLODataset(
        data_root / "images" / "val", data_root / "labels" / "val",
        imgsz=args.imgsz, augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    # Optimizer & Scheduler — SOURCE: engine/trainer.py L248-292
    nb = len(train_loader)
    accumulate = max(round(64 / args.batch), 1)
    weight_decay = hyp.weight_decay * args.batch * accumulate / 64
    optimizer = build_optimizer(model, lr=hyp.lr0, momentum=hyp.momentum, decay=weight_decay)

    if hyp.cos_lr:
        lf = lambda x: max((1 - math.cos(x * math.pi / args.epochs)) / 2, 0) * (hyp.lrf - 1) + 1
    else:
        lf = lambda x: max(1 - x / args.epochs, 0) * (1.0 - hyp.lrf) + hyp.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # AMP, EMA, Early stopping
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = ModelEMA(model)
    stopper = EarlyStopping(patience=args.patience)

    # Freeze DFL
    for k, v in model.named_parameters():
        if ".dfl" in k:
            v.requires_grad = False

    # CSV logger — includes val loss columns
    csv_path = save_dir / "results.csv"
    csv_header = ["epoch", "box_loss", "cls_loss", "dfl_loss", "val_box_loss", "val_cls_loss", "val_dfl_loss", "lr", "mAP50"]

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_fitness = 0.0
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            # Try default last.pt in save_dir
            ckpt_path = wdir / "last.pt"
        assert ckpt_path.exists(), f"Resume checkpoint not found: {ckpt_path}"
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Restore model weights (training model, not just EMA)
        model.load_state_dict(ckpt["train_model_state_dict"] if "train_model_state_dict" in ckpt else ckpt["model_state_dict"])
        ema.ema.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_fitness = ckpt.get("best_fitness", ckpt.get("fitness", 0.0))
        # Restore E2ELoss decay state
        if "e2e_updates" in ckpt:
            criterion.updates = ckpt["e2e_updates"]
            criterion.o2m = criterion.decay(criterion.updates)
            criterion.o2o = max(criterion.total - criterion.o2m, 0)
        if "ema_updates" in ckpt:
            ema.updates = ckpt["ema_updates"]
        print(f"Resumed at epoch {start_epoch}, best_fitness={best_fitness:.4f}")
        del ckpt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    # ---- Training loop — SOURCE: engine/trainer.py L376-584 ----
    nw = max(round(hyp.warmup_epochs * nb), 100)
    t0 = time.time()
    print(f"\nStarting training from epoch {start_epoch+1} to {args.epochs}...\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        scheduler.step()
        tloss = None
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            ni = i + nb * epoch

            # Warmup — SOURCE: engine/trainer.py L425-439
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, int(np.interp(ni, xi, [1, 64 / args.batch]).round()))
                for x in optimizer.param_groups:
                    x["lr"] = np.interp(ni, xi, [
                        hyp.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                        x["initial_lr"] * lf(epoch),
                    ])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp.warmup_momentum, hyp.momentum])

            # Forward + Loss — SOURCE: engine/trainer.py L442-456
            imgs = batch["img"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=(device.type == "cuda")):
                preds = model(imgs)
                loss, loss_items = criterion(preds, batch)
                total_loss = loss.sum()

            # Backward — SOURCE: engine/trainer.py L458-459
            scaler.scale(total_loss).backward()

            # Optimizer step — SOURCE: engine/trainer.py L739-747
            if (ni - (-1)) >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            # Logging
            tloss = loss_items if tloss is None else (tloss * i + loss_items) / (i + 1)
            if i % 20 == 0 or i == nb - 1:
                mem = f"{torch.cuda.memory_reserved(device) / 1e9:.1f}G" if device.type == "cuda" else "CPU"
                print(f"  Epoch {epoch+1}/{args.epochs}  Batch {i+1}/{nb}  "
                      f"Mem {mem}  box={tloss[0]:.4f}  cls={tloss[1]:.4f}  dfl={tloss[2]:.4f}")

        # Update E2ELoss decay — SOURCE: utils/loss.py L1186-1190
        criterion.update()

        # Validation (full: computes val loss via criterion)
        mAP50, vloss = validate(ema.ema, val_loader, device, args.nc, criterion=criterion)
        vloss = vloss if vloss is not None else torch.zeros(3, device=device)
        fitness = mAP50
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1} complete — mAP50={mAP50:.4f}  "
              f"val_box={vloss[0]:.4f}  val_cls={vloss[1]:.4f}  val_dfl={vloss[2]:.4f}  "
              f"lr={current_lr:.6f}")

        # Save CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{tloss[0]:.5f}", f"{tloss[1]:.5f}", f"{tloss[2]:.5f}",
                f"{vloss[0]:.5f}", f"{vloss[1]:.5f}", f"{vloss[2]:.5f}",
                f"{current_lr:.6f}", f"{mAP50:.5f}",
            ])

        # Save checkpoints — SOURCE: engine/trainer.py L642-687
        ckpt = {
            "epoch": epoch,
            "model_state_dict": ema.ema.state_dict(),
            "train_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "fitness": fitness,
            "best_fitness": best_fitness if fitness < best_fitness else fitness,
            "e2e_updates": criterion.updates,
            "ema_updates": ema.updates,
            "args": vars(hyp),
        }
        torch.save(ckpt, wdir / "last.pt")
        if fitness >= best_fitness:
            best_fitness = fitness
            torch.save(ckpt, wdir / "best.pt")
        print(f"  Saved checkpoints to {wdir}")

        # Early stopping
        if stopper(epoch, fitness):
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break

        # Memory cleanup
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nTraining complete. {epoch+1} epochs in {elapsed/3600:.2f} hours.")
    print(f"Best mAP50 = {best_fitness:.4f}")
    print(f"Results saved to {save_dir}")

    # Plot metrics
    plot_results(csv_path, save_dir)


# ============================= Main ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 Standalone Training")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset root (train/val/images/labels)")
    parser.add_argument("--nc", type=int, default=80, help="Number of classes")
    parser.add_argument("--scale", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="Model scale")
    parser.add_argument("--reg_max", type=int, default=1, help="DFL bins (1=no DFL, 16=full DFL)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="runs/yolo26", help="Save directory")
    parser.add_argument("--resume", type=str, default="", help="Path to last.pt checkpoint to resume from (or 'auto' for default)")
    args = parser.parse_args()
    # Shortcut: --resume auto → use default last.pt path
    if args.resume == "auto":
        default_ckpt = Path(args.save_dir) / "weights" / "last.pt"
        args.resume = str(default_ckpt) if default_ckpt.exists() else ""
    train(args)