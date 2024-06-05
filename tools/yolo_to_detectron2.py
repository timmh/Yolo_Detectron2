import os
import argparse
import torch


def yolov5_to_detectron2(yolov5_input_path, detectron2_output_path):
    
    # load yolov5 weights
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "yolov5"))
    m = torch.load(os.path.join(yolov5_input_path), map_location=torch.device("cpu"), weights_only=False)
    m = m["model"].state_dict()

    # save detectron2 weights
    torch.save(m, detectron2_output_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("yolov5_input_path", type=str, help="Path to yolov5 input weights")
    argparser.add_argument("detectron2_output_path", type=str, help="Path to detectron2 output weights")
    args = argparser.parse_args()
    yolov5_to_detectron2(args.yolov5_input_path, args.detectron2_output_path)