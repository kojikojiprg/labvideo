import sys

import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from mmengine.registry import DefaultScope
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
from mmyolo.utils import register_all_modules

sys.path.append("src")
from utils import yaml_handler


class YOLO:
    def __init__(self, cfg_path: dict, device: str):
        self._cfg = yaml_handler.load(cfg_path)
        self._device = device

        # build the detection model from a config file and a checkpoint file
        register_all_modules()
        mmdet_cfg = Config.fromfile(cfg_path)
        self._det_model = init_detector(
            mmdet_cfg.config, mmdet_cfg.weights, device=device
        )
        self._det_model.cfg = adapt_mmdet_pipeline(self._det_model.cfg)

    def __del__(self):
        del self._det_model

    def predict(self, img: np.array):
        with DefaultScope.overwrite_default_scope("mmyolo"):
            det_results = inference_detector(self._det_model, img)
        bboxes = self._process_det_results(
            det_results,
        )

        return bboxes

    def _process_det_results(self, det_result):
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None], pred_instance.labels.reshape(-1, 1)),
            axis=1,
        )
        bboxes = bboxes[pred_instance.scores > self._cfg.th_conf]
        bboxes = bboxes[nms(bboxes, self._cfg.th_iou)]
        return bboxes
