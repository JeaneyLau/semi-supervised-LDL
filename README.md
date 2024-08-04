# SS-ALDL: Consistency-Based Semi-Supervised Label Distribution Learning for Acne Severity Classification
This is the code repository for the Semi-Supervised Facial Acne Label Distribution Learning Model (SS-ALDL), implemented by Pytorch. SS-ALDL is the first semi-supervised label distribution learning model designed for facial acne grading tasks. It integrates the acne grading task and acne counting task using label distribution learning. Meanwhile, the semi-supervised training method enables it to learn features of unlabeled data. In addition, we propose a feature similarity learning loss to enhance consistency. All experiments were conducted on the ACNE04 public dataset.

Graphical Abstract:
<div align="center" >
  <img src="https://github.com/JeaneyLau/semi-supervised-LDL/blob/main/image/Abstract.jpg">
</div>

## Usage

### Clone or download this repo
```
git clone https://github.com/JeaneyLau/semi-supervised-LDL.git
```
### Dateset

**The ACNE04 dataset**. It is a public dataset consisting of 1457 facial acne images, each annotated with global acne severity grading and the local lesion numbers. The annotation of all images followed the Hayashi grading criteria. This dataset can be downloaded from https://github.com/xpwu95/ldl.

### Train and evaluation
```
python main.py
```
### Additional Information
After training, the result will be saved in `./result/SS_ALDL`. Among them, the training logs are saved in `./result/SS_LDL/train.csv`, and the test logs are saved in `./result/SS_LDL/test.csv`.

## Result

### 10% labeled data 

| Method | ACC| MAE | YI |
| ---- | -------| ----- |----|
| MeanTeacher|  0.681 | 0.342| 0.471 |
| SRC-MT|  0.684 | 0.342 | 0.490|
| MixMatch|  0.702 | 0.345|0.435 |
| ReMixMatch|  0.705 |0.315|0.529 |
| FixMatch|  0.691|0.339| 0.515|
| FlexMatch|  0.702 |0.311| 0.525|
| FreeMatch|  0.702 |0.315| 0.452|
| SoftMatch|  0.695 |0.311| *0.538* |
| SS-ALDL (Ours)|  *0.708* |*0.297*| 0.511|

### 20% labeled data 

| Method | ACC| MAE | YI |
| ---- | -------| ----- |----|
| MeanTeacher|  0.726 | 0.284| 0.555 |
| SRC-MT|  0.743 | 0.263 | 0.587|
| MixMatch|  0.712 | 0.304|0.474 |
| ReMixMatch|  0.749 |0.273|0.548 |
| FixMatch|  0.726|0.280| 0.559|
| FlexMatch|  0.736 |0.270| 0.599|
| FreeMatch|  0.739 |0.270| 0.582|
| SoftMatch|  0.743 |0.277| 0.591 |
| SS-ALDL (Ours)|  *0.756* |*0.246*| *0.605*|
