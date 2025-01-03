## :computer: Usage

### 0. Dataset Preparation

- The dataset for training can be downloaded here [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), and [RealSR](https://github.com/csjcai/RealSR?tab=readme-ov-file).

- The dataset for testing can be downloaded here [BasicSR.](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

- It is recommended to symlink the dataset root to `Datasets` with the follow command:
 
- The file structure is as follows:

  ```
  Data
  Datasets
  ├─Benchmark   
  │  ├─Set5
  │  │  ├─GTmod12
  │  │  ├─LRbicx2
  │  │  ├─LRbicx3
  │  │  ├─LRbicx4
  │  │  └─original
  │  ├─Set14
  │  ├─BSDS100
  │  ├─Manga109
  │  └─Urban100
  ├─DF2K
  │  ├─DF2K_HR_train
  │  ├─DF2K_HR_train_x2m
  │  ├─DF2K_HR_train_x3m
  │  └─DF2K_HR_train_x4m  
  Demo
  ...
  ```

### 1. Evaluation
Download the pretrained weights here and run the following command for evaluation on five widely-used Benchmark datasets.
Remember to change the path in the file to your local path.
```
python Demo/infer.py 
```

### 2. Training
Please refer to the experimental setup of the paper for training details.

### 3. Source

The pretrained weight is here. (https://pan.baidu.com/s/1aE-uStecK1E5daRSSH0rGw ) (code:ecyk)  (https://drive.google.com/file/d/12Xi2oD4RvCiYjKoMpHAatvyAvTS0vwWJ/view?usp=sharing)

Visual results for the x4 are also included!

## 4. pray: Citation
If this work is helpful for you, please consider citing:

```
@article{wang2024efficient,
  title={Efficient image super resolution via Mixed Window and Dimension Interaction},
  author={Wang, Shouyi and Liu, Gang and Liu, Xiao and Liao, Xiangyu and Ren, Chao},
  journal={Neurocomputing},
  pages={129211},
  year={2024},
  publisher={Elsevier}
}
```
