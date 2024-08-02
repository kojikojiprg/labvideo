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

options:
- ```-f, --finetuned_model```: finetuningされたyolov8のweightsを使用する
- ```-v, --video```: 検出結果のmp4を保存する

### yolov8_finetuning.py
YOLOv8を実験器具でファインチューニングする  
```annotation/yolov8_finetuning/``` にあるデータから学習用とテスト用のデータセットに分け、```datasets/yolov8_finetuning/``` に保存する  
このデータセットを用いて学習する  

options:
- ```-cd, --create_dataset```: データセットを作成する


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
object_tracking.py の結果に対して、LLaVAを用いて物体にラベルをつける
```out/[動画名]/[動画名]_llava.tsv``` に結果を保存

### count_paint_within_bbox.py
YOLOv8の予測結果の中に、Paintの中心座標がどれくらいの入っているかを計算する
```out/count_patin_within_bbox.tsv``` に結果を保存

### classify_yolo.py
Yolov8n-cls.pt をファインチューニングして分類した  
positional arguments:
- ```dataset_type```: 'paint' or 'yolo'
  - Paintで囲まれたエリアの画像の分類を行う
  - 物体認識結果 ```out/[動画名]/[動画名]_det.tsv``` で囲まれたエリアの画像の分類を行う
- ```data_type```: 'label' or 'label_type'
  - 'label': A11~C42
  - 'label_type': A~C
  - 'anomaly'(dataset_type==yolo): 0(異常なし), 1(異常あり)
  dataset==yoloの時、Paintの±th_sec秒以内でIoU>th_iouのデータを異常データ、それ以外を異常なしデータとする

options:
- ```-cd, --create_dataset```: データセットを作成する
- ```-tr, --train```: ファインチューニングを行う
- ```-v, --version```: テストバージョン(--train を指定したときは無効)
- ```th_sec```(dataset==yolo): 異常とするYOLOの物体認識結果のPaintとの発生時間の閾値
- ```th_iou```(dataset==yolo): 異常とするYOLOの物体認識結果のPaintとのIoUの閾値

#### アップデート履歴
- v1: ラベル毎に学習データとテストデータを分類 (データリークを防ぐため)
