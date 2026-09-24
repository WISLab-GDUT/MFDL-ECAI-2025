# MFDL: Frequency-Domain Deepfake Detection (ECAI 2025)

Source code of the MFDL model for detecting AI-generated (deepfake) images in the
frequency domain. Each stage of the backbone pairs two branches and fuses them:

* **FRE / wavelet branch** — DWT decomposition, high-frequency sub-bands rescaled,
  inverse DWT, then a point-wise convolution (`dwt` + `idwt`, `weightdw*`).
* **FRC / Fourier branch** — high-frequency whitening in the 2-D FFT domain
  (`hfreqWH`), channel-wise high-frequency filtering (`hfreqC`) and a complex-valued
  convolution (`ComplexConv2d`) before the inverse FFT.

## Repository layout

```
networks/MFDL.py        model definition (MFDL, mfdl() factory)
networks/base_model.py  base nn.Module wrapper (optimizer, LR schedule, checkpoint IO)
networks/MFDLtrainer.py training wrapper (BCEWithLogitsLoss, train step)
MFDLtrain.py            training entry point
MFDLtest.py             evaluation entry point (per test folder acc / AP)
validate.py             accuracy / average precision on one dataset
test.py, testdataloader.py  image-level test with JPEG / blur robustness options (writes CSV)
eval.py                 evaluation loop over dataset groups
data/                   ImageFolder-based dataset + dataloader (binary mode)
options/                command line options (train / test)
util.py                 logger and helpers
weights/MFDL.pth        released checkpoint
```

## Environment

Python 3.10, PyTorch 2.6 (CUDA) — see `requirements.txt`:

```
pip install -r requirements.txt
```

## Data layout

Binary mode expects an `ImageFolder` layout, real images in `0_real`, fake in `1_fake`:

```
<dataroot>/<split>/0_real/*.png
<dataroot>/<split>/1_fake/*.png
```

Images are resized to `--loadSize` (256) and center-cropped to `--cropSize` (224)
during evaluation, with ImageNet normalization.

## Training

```
python MFDLtrain.py --dataroot /path/to/dataset --train_split train --name mfdl
```

Checkpoints are written as `MFDL{epoch}.pth` in the working directory
(`--continue_train` reloads `--epoch`).

## Evaluation

```
python MFDLtest.py --model_path weights/MFDL.pth
```

`DetectionTests` in `MFDLtest.py` lists the test roots to iterate over
(one sub-folder per generator); each folder should contain `0_real` / `1_fake`.
For a single dataset:

```
python validate.py --model_path weights/MFDL.pth --dataroot /path/to/test_root --no_resize
```

## Pretrained weights

`weights/MFDL.pth` is the trained MFDL checkpoint used for evaluation.

