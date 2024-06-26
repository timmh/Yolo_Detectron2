"""
Yolo Training script
This script is a simplified version of the script in detectron2/tools
"""

from pathlib import Path
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch
)
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg

from yolo_detectron2 import add_yolo_config


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = Path(cfg.OUTPUT_DIR) / "inference"
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[])

        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).save("yolov5m-detectron2")
    del model

    trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    trainer.checkpointer.save('yolov5m-detectron2')
    trainer.run_step()
    trainer.checkpointer.save('yolov5m-detectron2')


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
