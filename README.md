# tx2-erfpsp
This repo is a tensorflow-version ERF-PSPNET implementation running on Nvidia Jetson TX2, based on the papers below:
- [Unifying Terrain Awareness for the Visually Impaired through Real-Time Semantic Segmentation](https://www.mdpi.com/1424-8220/18/5/1506)
- [Unifying terrain awareness through real-time semantic segmentation](http://www.wangkaiwei.org/file/publications/iv2018_kailun.pdf)

We use ZED Mini ZED Mini Stereo Camera to capture the road scene.

## Prerequisites
- Jetpack 3.1 (CUDA 8.0/cuDNN 6.0)
- Python 3.5
- ZED SDK 2.3

## Requirements for Python
- TensorFlow 1.4 (need to be compiled on TX2)
- cv2
- pyzed

## Usage
```python
python3 main.py
```