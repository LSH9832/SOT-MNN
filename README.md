# SOT-MNN: Single Object Tracking MNN Deployment

- this project converts python inference code([pysot](https://github.com/STVIR/pysot)) to cpp using MNN SDK.
- [MNN models](https://github.com/LSH9832/SOT-MNN/releases/download/v0.0.1/weights.zip) are directly available.

![](assets/demo.jpg)

## setup

```sh
git clone https://github.com/LSH9832/SOT-MNN
cd SOT-MNN
python setup.py
```

## run

```sh
# view argparse help
./build/mnn_det -?

# example
./build/mnn_det --cfg config/alex.yaml \        # config file
                --source /Path2yourVideo.mp4 \  # your video file or rtmp/rtsp stream or camera device such as /dev/video0
                --pause                         # pause at start, press space key to start
```