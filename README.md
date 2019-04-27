# tx2-erfpsp
This repo is a tensorflow-version ERF-PSPNET implementation running on NVIDIA Jetson TX2, based on the papers below:
- [Unifying Terrain Awareness for the Visually Impaired through Real-Time Semantic Segmentation](https://www.mdpi.com/1424-8220/18/5/1506)
- [Unifying terrain awareness through real-time semantic segmentation](http://www.wangkaiwei.org/file/publications/iv2018_kailun.pdf)

We use ZED Mini Stereo Camera to capture the road scene.

## Prerequisites
- Jetpack 3.1 (CUDA 8.0/cuDNN 6.0)
- Python 3.5
- ZED SDK 2.3

## Requirements for Python
- TensorFlow 1.4 (need to be compiled on TX2)
- cv2
- pyzed
- opencv-python

## Usage
```python
python3 main.py
```

FYI, because the SpaceToBatchND op of TensorFlow has incorrect output on CUDA 9 (see https://devtalk.nvidia.com/default/topic/1044411/jetson-tx2/tensorflow-op-spacetobatchnd-does-not-work-correctly-on-tx2/post/5333429/), the code could only run with CUDA 8.0 (such that the only choice is Jetpack 3.1). Accordingly, the highest version that could be compiled smoothly on TX2 is tf 1.4.
