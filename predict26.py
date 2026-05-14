"""YOLO26 standalone inference script.
Runs a trained YOLO26 model on an image and draws the bounding boxes.

Usage on Colab:
    !python predict26.py --weights runs/yolo26/weights/best.pt --source test_image.jpg --conf 0.25
"""
import argparse
import os
import time
import cv2
import numpy as np
import torch

from yolo26_modules import build_yolo26


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


def draw_boxes(image, boxes, class_names=None):
    """Draw bounding boxes on image."""
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        
        # Color based on class ID
        color = ((cls_id * 50) % 255, (cls_id * 100) % 255, (cls_id * 150) % 255)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"Class {cls_id} {conf:.2f}"
        if class_names and cls_id < len(class_names):
            label = f"{class_names[cls_id]} {conf:.2f}"
            
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Load checkpoint
    print(f"Loading weights from {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    
    # Extract model args
    hyp_args = ckpt.get("args", {})
    nc = hyp_args.get("nc", args.nc)
    scale = hyp_args.get("scale", "n")
    reg_max = hyp_args.get("reg_max", 1)

    # Build model and load weights
    model = build_yolo26(nc=nc, scale=scale, reg_max=reg_max)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Prepare image
    img0 = cv2.imread(args.source)
    assert img0 is not None, f"Image not found: {args.source}"
    
    # Preprocess
    img, ratio, (dw, dh) = letterbox(img0, new_shape=args.imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Inference
    print("Running forward pass...")
    t0 = time.time()
    with torch.no_grad():
        # model returns (y, preds) in eval mode. y is [bs, max_det, 6]
        y, _ = model(img)
    t1 = time.time()
    print(f"Inference time: {(t1 - t0) * 1000:.1f} ms")

    # Postprocess
    pred = y[0] # first image in batch
    pred = pred[pred[:, 4] > args.conf] # filter by confidence
    
    # Scale boxes back to original image size
    if len(pred):
        pred[:, 0] -= dw  # x1
        pred[:, 1] -= dh  # y1
        pred[:, 2] -= dw  # x2
        pred[:, 3] -= dh  # y2
        pred[:, :4] /= ratio
        
        # Clip to image bounds
        pred[:, [0, 2]] = pred[:, [0, 2]].clamp(0, img0.shape[1])
        pred[:, [1, 3]] = pred[:, [1, 3]].clamp(0, img0.shape[0])
    
    print(f"Detected {len(pred)} objects.")
    
    # Draw and save
    draw_boxes(img0, pred.cpu().numpy())
    
    out_name = f"pred_{os.path.basename(args.source)}"
    cv2.imwrite(out_name, img0)
    print(f"Saved result to {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 Standalone Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--nc", type=int, default=80, help="Fallback class count if not in ckpt")
    args = parser.parse_args()
    predict(args)
