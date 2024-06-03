# labvideo
## Environments
- CUDA 11.8
- Python 3.10.14

## Installation
```
git submodule --init
conda env -f env.yml
pip install -U torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
```


## Scripts
### object_tracking.py
YOLOとSMILEtrackを使用して物体認識とトラッキングを行う  
```out/[動画名]/[動画名]_det.tsv``` に結果を保存
```out/[動画名]/[動画名]_det.mp4``` に結果の動画を保存

### get_paint_bbox.py
```annotation/video/``` にある動画の黒丸部分を切り出す
- ```annotation/paint_bbox.json``` に切り出した結果の座標を保存
- ```annotation/paint_error.tsv``` に切り出せなかった結果を保存
- ```annotation/images/``` に切り出した画像を保存

### collect_annotation.py
```annotation/annotation.json``` と ```annotation/paint_bbox.json``` からアノテーションを時系列に並べる  
```out/[動画名]/[動画名]_ann.tsv``` に結果を保存

### compare_ann_det.py
```out/[動画名]/[動画名]_ann.tsv``` と ```out/[動画名]/[動画名]_det.tsv``` の結果を3Dグラフにする  
```out/compare_ann_det/[動画名]_plot.mp4``` に3Dグラフを縦軸で回転させた動画を保存　

### create_trainsition_matrix.py
```annotation/annotation.json``` からラベルの遷移行列を生成する  
```out/transition_matrix/``` に遷移行列のプロットを保存  
```*_clipXX.jpg``` は状態遷移をXXを上限にしたプロット

### predict_llava.py
object_tracking.py の結果に対して、LLaVAを用いて物体にラベルをつける(実装中)

### count_paint_within_bbox.py
YOLOv8の予測結果の中に、Paintの中心座標がどれくらいの入っているかを計算する
```out/count_patin_within_bbox.tsv``` に結果を保存
