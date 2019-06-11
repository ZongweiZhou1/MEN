train code for `A Unified Multi-cue Extraction Network for Online Multi-Target Tracking`.
---

### prerequisites
* Python 3.6
* Pytorch 0.3.1
* CUDA 0.8 or higher

### Build

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, PSROI_POOLING, ROI_Pooing, ROI_Align and ROI_Crop. 
The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**