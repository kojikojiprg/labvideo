# labvideo
## installation
```
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.0.1 mmdet mmpose mmyolo
```

```
git submodule --init
cd submodules/llava/
pip install -e .

# downgrade torch for cuda11.8
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118
```
