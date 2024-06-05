import logging
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn, Tensor
from typing import List, Dict, Tuple
from types import MethodType

from detectron2.config import configurable
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.postprocessing import detector_postprocess

from ..yolov5.models.yolo import parse_model

from .general import non_max_suppression
from .loss import ComputeLoss


__all__ = ["Yolo"]

logger = logging.getLogger(__name__)


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N,Ai,H,W,K) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 5, tensor.shape
    N = tensor.shape[0]
    tensor = tensor.view(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def head_forward(self, x, run_inference=False):
    z = []  # inference output
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if run_inference:  # inference
            if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))
    
    return (x, torch.cat(z, 1) if run_inference else None)


@META_ARCH_REGISTRY.register()
class Yolo(nn.Module):
    """
    Implement YoloV5
    """

    @configurable
    def __init__(
        self,
        *,
        model: nn.Module,
        save,
        loss,
        num_classes,
        conf_thres,
        iou_thres,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="RGB",
    ):
        super().__init__()

        self.model = model
        self.save = save
        self._size_divisibility = 32  # TODO: infer this from self.model instead of hard-coding
        self.model[-1].forward = MethodType(head_forward, self.model[-1])  # overwrite forward method of head

        self.num_classes = num_classes
        self.single_cls = num_classes == 1
        # Inference Parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        self.loss = loss

        self.init_stride()

    @classmethod
    def from_config(cls, cfg):
        model_yaml_file = cfg.MODEL.YAML
        if model_yaml_file.startswith("yolov5://"):
            model_yaml_file = model_yaml_file.replace("yolov5:/", str((Path(__file__) / ".." / ".." / "yolov5" / "models").resolve()))
        import yaml  # for torch hub
        with open(model_yaml_file) as f:
            model_yaml = yaml.safe_load(f)  # model dict
        model_yaml['nc'] = cfg.MODEL.YOLO.NUM_CLASSES
        model, save = parse_model(model_yaml, [len(cfg.MODEL.PIXEL_MEAN)])
        head = model[-1]

        loss = ComputeLoss(cfg, head)
        return {
            "model": model,
            "save": save,
            "loss": loss,
            "num_classes": head.nc,
            "conf_thres": cfg.MODEL.YOLO.CONF_THRESH,
            "iou_thres": cfg.MODEL.YOLO.IOU_THRES,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        # TODO: find out why below assertion fails and fix it instead of returning early
        if len(batched_inputs) != len(results):
            print(f"WARNING: Failed to run visualize_training because inputs and results are of different sizes ({len(batched_inputs)}, {len(results)})")
            return

        # assert len(batched_inputs) == len(
        #     results
        # ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def init_stride(self):
        s = 256  # 2x min stride
        dummy_input = torch.zeros(1, len(self.pixel_mean), s, s)
        pred, _ = self._forward_once(dummy_input)
        self.model[-1].stride = torch.tensor(
            [s / x.shape[-2]
                for x in pred])  # forward
        self.model[-1].anchors /= self.model[-1].stride.view(-1, 1, 1)
        self.stride = self.model[-1].stride
        self._initialize_biases()  # only run once
        self.loss._initialize_ssi(self.stride)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.model[-1].m, self.model[-1].stride):  # from
            b = mi.bias.view(self.model[-1].na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + self.model[-1].nc] += math.log(0.6 / (self.model[-1].nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _forward_once(self, x, run_inference=False):
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x) if m != self.model[-1] else m(x, run_inference=run_inference)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        run_visualize_training = self.vis_period > 0 and get_event_storage().iter % self.vis_period == 0
        pred, z = self._forward_once(images.tensor, run_inference=run_visualize_training)
            
        assert not torch.jit.is_scripting(), "Not supported"
        assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        losses = self.loss(pred, gt_instances, images)

        if run_visualize_training:
            results = self.process_inference(
                z, images.image_sizes
            )
            self.visualize_training(batched_inputs, results)

        return losses

    def inference(self, batched_inputs: Tuple[Dict[str, Tensor]], do_postprocess=True):
        images = self.preprocess_image(batched_inputs)
        _, z = self._forward_once(images.tensor, run_inference=True)
        results = self.process_inference(z, images.image_sizes)
        if torch.jit.is_scripting() or not do_postprocess:
            return results
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def process_inference(self, out, image_sizes):
        out = non_max_suppression(out, self.conf_thres, self.iou_thres, multi_label=True, agnostic=self.single_cls)
        assert len(out) == len(image_sizes)
        results_all: List[Instances] = []
        # Statistics per image
        for si, (pred, img_size) in enumerate(zip(out, image_sizes)):

            if len(pred) == 0:
                continue

            # Predictions
            if self.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # Predn shape [ndets, 6] of format [xyxy, conf, cls] relative to the input image size
            result = Instances(img_size)
            result.pred_boxes = Boxes(predn[:, :4])  # TODO: Check if resizing needed
            result.scores = predn[:, 4]
            result.pred_classes = predn[:, 5].int()   # TODO: Check the classes
            results_all.append(result)
        return results_all

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self._size_divisibility)
        return images