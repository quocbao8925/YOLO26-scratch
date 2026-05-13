# YOLO26-scratch
--- 
Upload all 3 files to Colab, then:
```python
!python train26.py \
    --data /content/dataset \
    --nc 1 \
    --scale n \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```
Dataset structure required:
```python
dataset/
  train/images/   train/labels/
  val/images/     val/labels/
```
