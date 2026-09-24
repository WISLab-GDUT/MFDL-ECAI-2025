
# MFDL: Multi-Perspective Frequency Domain Learning for Generalizable AI-Generated Image Detection

Official code for the ECAI 2025 paper:

> **Multi-Perspective Frequency Domain Learning for Generalizable AI-Generated Image Detection**
> Zili Xu, Fuqiang Yu, Jianjie Luo, Zhenguo Yang
> *ECAI 2025 — Frontiers in Artificial Intelligence and Applications, Vol. 413, IOS Press*
> DOI: [10.3233/FAIA251048](https://doi.org/10.3233/FAIA251048)

Affiliations: Guangdong University of Technology · Key Laboratory of Computing Power Network and
Information Security, Ministry of Education, Qilu University of Technology (Shandong Academy of Sciences).
Corresponding author: jianjieluo@gdut.edu.cn

## Overview

Most frequency-based detectors extract forgery traces with the Fast Fourier Transform (FFT) alone, which
does not capture a comprehensive frequency-domain representation of AI-generated images. **MFDL** learns
both generalized and discriminative frequency representations by combining the **Discrete Wavelet
Transform (DWT)** and the **FFT** in a dual-branch architecture:
<img width="715" height="365" alt="截屏2026-09-24 18 54 06" src="https://github.com/user-attachments/assets/d8ad1d07-4b7f-4394-8123-78812e17617b" />


## Repository layout

```
networks/MFDL.py        MFDL model (FRE + FRC branches, fusing backbone), mfdl() factory
networks/base_model.py  base nn.Module wrapper (optimizer, LR schedule, checkpoint IO)
networks/MFDLtrainer.py training wrapper (BCEWithLogitsLoss, train step)
MFDLtrain.py            training entry point
MFDLtest.py             evaluation entry point (per test folder Acc / AP)
validate.py             accuracy / average precision on one dataset
test.py, testdataloader.py  image-level test with JPEG / blur robustness options (writes CSV)
eval.py                 evaluation loop over dataset groups
data/                   ImageFolder-based dataset + dataloader (binary mode)
options/                command line options (train / test)
util.py                 logger and helpers
weights/MFDL.pth        released checkpoint
```

## Environment

Python 3.10 with PyTorch 2.6 (CUDA) was used for the experiments:

```
pip install -r requirements.txt
```

## Data layout

Binary mode expects an `ImageFolder` layout — real images in `0_real`, generated images in `1_fake`:

```
<dataroot>/<split>/0_real/*.png
<dataroot>/<split>/1_fake/*.png
```

Images are resized to `--loadSize` (256) and center-cropped to `--cropSize` (224) at evaluation time,
with ImageNet normalization.

## Training

The paper trains on the four-class ProGAN subset of ForenSynths (car, cat, chair, horse; >70k images)
for 65 epochs with Adam (lr 1e-3, batch size 32):

```
python MFDLtrain.py --dataroot /path/to/ProGAN_4class \
                    --train_split train \
                    --optim adam --lr 0.001 --batch_size 32 --niter 65 \
                    --name mfdl
```

Checkpoints are written as `MFDL{epoch}.pth` in the working directory; `--continue_train --epoch <n>`
resumes from one. The learning rate is decayed by 0.8 every `--delr_freq` (20) epochs.

## Evaluation

Following the baselines, results are reported as accuracy (Acc) and average precision (AP):

```
python MFDLtest.py --model_path weights/MFDL.pth
```

`DetectionTests` in `MFDLtest.py` lists the test roots to iterate over (one sub-folder per generator);
each folder must contain `0_real` / `1_fake`. For a single dataset:

```
python validate.py --model_path weights/MFDL.pth \
                   --dataroot /path/to/test_root --no_resize
```

## Pretrained weights

`weights/MFDL.pth` is the released checkpoint (2.0M parameters, 8.0 MB).

## Results (paper)

Mean Acc / AP over each benchmark group; MFDL is trained on ProGAN only.

| Benchmark | Test generators | Mean Acc | Mean AP |
| --- | --- | --- | --- |
| GAN | CycleGAN, StyleGAN, StyleGAN2, BigGAN, ProGAN, StarGAN, AttGAN, RelGAN (8) | 96.5 | 99.6 |
| Diffusion | DALL·E, LDM200, LDM100, PNDM, DDPM, Glide, SDv4, VQDiffusion, Guided (9) | 92.6 | 92.1 |
| DiffusionForensics | ADM, DDPM, IDDPM, LDM, PNDM, VQDiffusion, SDv1, SDv2 (8) | 95.1 | 99.6 |
| GenImage | BigGAN, Wukong, VQDM, Glide, Midjourney, ADM, SDv5 (7) | 88.1 | 95.5 |



## Citation

```bibtex
@inproceedings{xu2025mfdl,
  title     = {Multi-Perspective Frequency Domain Learning for Generalizable AI-Generated Image Detection},
  author    = {Xu, Zili and Yu, Fuqiang and Luo, Jianjie and Yang, Zhenguo},
  booktitle = {Proceedings of the European Conference on Artificial Intelligence (ECAI 2025)},
  series    = {Frontiers in Artificial Intelligence and Applications},
  volume    = {413},
  publisher = {IOS Press},
  year      = {2025},
  doi       = {10.3233/FAIA251048}
}
```

## Notes on this release

This repository contains the model, training and evaluation code, together with the released
checkpoint. The analysis/visualisation scripts behind Figures 4–6 of the paper (Grad-CAM, t-SNE,
frequency-energy distributions) and the robustness image generator are not part of this release.
