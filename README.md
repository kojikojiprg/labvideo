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
### Annotation
#### get_paint_bbox.py
```annotation/video/``` にある動画の黒丸部分を切り出す
- ```annotation/paint_bbox.json``` に切り出した結果の座標を保存
- ```annotation/paint_error.tsv``` に切り出せなかった結果を保存
- ```annotation/images/``` に切り出した画像を保存

#### collect_annotation.py
```annotation/annotation.json``` と ```annotation/paint_bbox.json``` からアノテーションを時系列に並べる  
```out/[動画名]/[動画名]_ann.tsv``` に結果を保存

#### create_trainsition_matrix.py
```annotation/annotation.json``` からラベルの遷移行列を生成する  
```out/transition_matrix/``` に遷移行列のプロットを保存  
```*_clipXX.jpg``` は状態遷移をXXを上限にしたプロット

### Prediction
#### yolov8_finetuning.py
YOLOv8を実験器具でファインチューニングする  
```annotation/yolov8_finetuning/``` にあるデータから学習用とテスト用のデータセットに分け、```datasets/yolov8_finetuning/``` に保存する  
このデータセットを用いて学習する  

options:
- ```-cd, --create_dataset```: データセットを作成する

#### object_tracking.py
YOLOとSMILEtrackを使用して物体認識とトラッキングを行う  
```out/[動画名]/[動画名]_det.tsv``` に結果を保存
```out/[動画名]/[動画名]_det.mp4``` に結果の動画を保存

options:
- ```-f, --finetuned_model```: finetuningされたyolov8のweightsを使用する
- ```-v, --video```: 検出結果のmp4を保存する

#### predict_llava.py
object_tracking.py の結果に対して、LLaVAを用いて物体にラベルをつける
```out/[動画名]/[動画名]_llava.tsv``` に結果を保存

options:
- ```-f, --finetuned_model```: finetuningされたyolov8の結果を使用する  
  ```out/[動画名]/[動画名]_llava_finetuned.tsv``` に結果を保存
- ```-pv, --prompt_version```: ```prompts/```からプロンプトを選択  

#### predict_llava_topk.py
object_tracking.py の結果のうち、各トラッキングIDの信頼度が高いK枚の画像に対して、LLaVAを用いて物体にラベルをつける
```out/[動画名]/[動画名]_llava.tsv``` に結果を保存

options:
- ```-f, --finetuned_model```: finetuningされたyolov8の結果を使用する  
  ```out/[動画名]/[動画名]_llava_finetuned.tsv``` に結果を保存
- ```-pv, --prompt_version```: ```prompts/```からプロンプトを選択  
- ```-ni, --n_imgs```: 指定した枚数だけ信頼度の高い画像を取得する  

#### predict_llava_finetuning_dataset.py
```datasets/yolov8_finetuning/```のデータに対して、LLaVAを用いて物体にラベルをつける
```out/[動画名]/[動画名]_llava.tsv``` に結果を保存

options:
- ```-f, --finetuned_model```: finetuningされたyolov8の結果を使用する  
  ```out/[動画名]/[動画名]_llava_finetuned.tsv``` に結果を保存
- ```-pv, --prompt_version```: ```prompts/```からプロンプトを選択  

#### compare_ann_det.py
```out/[動画名]/[動画名]_ann.tsv``` と ```out/[動画名]/[動画名]_det.tsv``` の結果を3Dグラフにする  
```out/compare_ann_det/[動画名]_plot.mp4``` に3Dグラフを縦軸で回転させた動画を保存　

options:
- ```-f, --finetuned_model```: finetuningされたyolov8の結果を使用する  
  ```out/compare_ann_det_finetuned/[動画名]_plot_finetuned.mp4``` に3Dグラフを縦軸で回転させた動画を保存　

#### count_paint_within_bbox.py
YOLOv8の予測結果の中に、Paintの中心座標がどれくらいの入っているかを計算する
```out/count_patin_within_bbox.tsv``` に結果を保存

options:
- ```-f, --finetuned_model```: finetuningされたyolov8の結果を使用する  
  ```out/count_patin_within_bbox_finetuned.tsv``` に結果を保存

#### plot_bbox_anomaly_labels.py
YOLOv8の検出結果が異常あり/異常なしをプロットする。  
options:
- ```-f, --finetuned_model```: finetuningされたyolov8の結果を使用する  
- ```th_sec```: 異常とするYOLOの物体認識結果のPaintとの発生時間の閾値
- ```th_iou```: 異常とするYOLOの物体認識結果のPaintとのIoUの閾値
- ```bbox_ratio```: Bboxをリサイズするためのフレームサイズに対しての比率

#### classify_paint.py
Yolov8n-cls.pt をファインチューニングしてアノテーション動画のPaint(丸で囲まれた部分)を分類した  
positional arguments:
- ```data_type```: 'label' or 'label_type'
  - 'label': A11~C42
  - 'label_type': A~C

options:
- ```-cd, --create_dataset```: データセットを作成する
- ```-tr, --train```: ファインチューニングを行う
- ```-v, --version```: テストバージョン(--train を指定したときは無効)

#### classify_yolov8.py
Yolov8n-cls.pt をファインチューニングして、アノテーション動画のPaintと重なるYOLO検出結果bboxを分類した  
positional arguments:
- ```data_type```: 'label' or 'label_type'
  - 'label': A11~C42
  - 'label_type': A~C
  dataset==yoloの時、Paintの±th_sec秒以内でIoU>th_iouのデータを異常データ、それ以外を異常なしデータとする

- ```split_type```: 'random' or 'video'
  学習データとテストデータの分け方
  - 'random': 全データでランダムに分割
  - 'video': 動画ごとに分割(yolov8_fintuning.pyのデータセットと同じように分割)

options:
- ```-f, --finetuned_model```: finetuningされたyolov8の結果を使用する  
- ```-cd, --create_dataset```: データセットを作成する
- ```-tr, --train```: ファインチューニングを行う
- ```-v, --version```: テストバージョン(--train を指定したときは無効)
- ```th_sec```(dataset==yolo): 異常とするYOLOの物体認識結果のPaintとの発生時間の閾値
- ```th_iou```(dataset==yolo): 異常とするYOLOの物体認識結果のPaintとのIoUの閾値

#### anomaly_detection.py
scikit-learn の OneClassSVM を用いて、YOLOv8の検出結果の異常検知を行った  
- 異常あり: アノテーション動画のPaintと重なるYOLOv8の検出結果
- 異常あり: アノテーション動画のPaintと重ならないYOLOv8の検出結果

positional arguments:
- ```split_type```: 'random' or 'video'
  学習データとテストデータの分け方
  - 'random': 全データでランダムに分割
  - 'video': 動画ごとに分割(yolov8_fintuning.pyのデータセットと同じように分割)

options:
- ```-f, --finetuned_model```: finetuningされたyolov8の結果を使用する  
- ```-cd, --create_dataset```: データセットを作成する
- ```-tr, --train```: ファインチューニングを行う
- ```-v, --version```: テストバージョン(--train を指定したときは無効)
- ```th_sec```(dataset==yolo): 異常とするYOLOの物体認識結果のPaintとの発生時間の閾値
- ```th_iou```(dataset==yolo): 異常とするYOLOの物体認識結果のPaintとのIoUの閾値
