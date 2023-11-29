import os
import os.path as osp
import sys

sys.path = [osp.abspath(osp.dirname(osp.dirname(__file__)))] + sys.path

from pysot.pysot.models.model_builder import ModelBuilder, ExportModelBuilder, ExportModelBuilder2
from pysot.pysot.core.config import cfg
from pysot.pysot.tracker.siamrpn_tracker import TensorRTTracker, TensorRTTracker2

import argparse
import torch
import onnx

import scripts.onnx2mnn as onnx2mnn

PROJ_ROOT = osp.dirname(osp.dirname(__file__))


def get_args():
    parser = argparse.ArgumentParser("SiamRPN export parser")

    parser.add_argument("--name", type=str, default="siamrpn_alex_dwxcorr")
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--mnn-root", type=str, default="/home/lsh/code/mnn_dev/MNN-2.7.1")
    parser.add_argument("--no-fp16", action="store_true")

    return parser.parse_args()


def main_export(args):

    weights = osp.join(PROJ_ROOT, f"weights/{args.name}/pth/model.pth")
    config = osp.join(PROJ_ROOT, f"weights/{args.name}/config.yaml")
    dirname = osp.join(PROJ_ROOT, f"weights/{args.name}/onnx")

    os.makedirs(dirname, exist_ok=True)

    assert osp.isfile(config), "config file does not exist!"

    cfg.merge_from_file(config)

    exporter = ExportModelBuilder2 if cfg.RPN.TYPE == 'MultiRPN' else ExportModelBuilder
    tracker = TensorRTTracker2 if cfg.RPN.TYPE == 'MultiRPN' else TensorRTTracker

    track = tracker(None, None, None)
    anchor = track.anchors
    model = exporter(state_dict=torch.load(weights, map_location="cpu"), anchor=anchor)

    template = torch.ones([1, 3, cfg.TRACK.EXEMPLAR_SIZE, cfg.TRACK.EXEMPLAR_SIZE])
    x = torch.ones([1, 3, cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE])

    model.start_export(template, x, dirname, True, True)

    onnx2mnn.convert(args.mnn_root, args.name, args.no_fp16)


def main():
    args = get_args()
    main_export(args)


if __name__ == '__main__':
    main()




