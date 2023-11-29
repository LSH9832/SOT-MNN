import os
import os.path as osp
import argparse
from glob import glob


PROJ_ROOT = osp.dirname(osp.dirname(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnn-root", type=str, default="/home/lsh/code/mnn_dev/MNN-2.7.1")
    parser.add_argument("--name", type=str, default="siamrpn_alex_dwxcorr")
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def convert(mnn_root, name, no_fp16):
    onnx_dir = osp.join(PROJ_ROOT, f"weights/{name}/onnx")
    mnn_dir = osp.join(PROJ_ROOT, f"weights/{name}/mnn")
    os.makedirs(mnn_dir, exist_ok=True)

    for onnx_file in glob(osp.join(onnx_dir, "*.onnx")):
        mnn_file_name = osp.basename(onnx_file)[:-4] + "mnn"
        command = f"{osp.join(mnn_root, 'build/MNNConvert')} -f ONNX --modelFile {onnx_file} " \
                  f"--MNNModel {osp.join(mnn_dir, mnn_file_name)} {'' if no_fp16 else '--fp16'}"

        print(command)
        os.system(command)


if __name__ == "__main__":
    args = get_args()
    convert(args.mnn_root, args.name, args.no_fp16)
