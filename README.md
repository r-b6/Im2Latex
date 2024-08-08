## Overview
An implementation of the model proposed by [Translating Math Formula Images to LaTeX Sequences Using Deep
Neural Networks with Sequence-level Training](https://arxiv.org/pdf/1908.11415) based off of [this](https://github.com/luopeixiang/im2latex) implementation of an older model by 
@luopeixiang. The model takes in images of mathematical formulas as input and outputs their corresponding LaTeX markup using a CNN encoder and LSTM decoder. 

## Performance

| BLEU-4  | Edit Distance | Exact Match |
| ------------- | ------------- | ------ |
| 0.8932  | 0.9226  | 0.3401 |

## Sample Results

![hero_image 001](https://github.com/user-attachments/assets/6cf9fa0a-c524-4184-8543-7ebbe2d26a92)


## Data and Preprocessing

The dataset containing images and their corresponding ground-truth sequences was sourced from [here](https://untrix.github.io/i2l/). Preprocessing was done through the process described
in [this](https://github.com/untrix/im2latex/tree/master/src/preprocessing) repository by @untrix. 

## Steps for use

Only follow steps 2-3 if you want to train a custom model instead of using the one in the latest release. 

### Install Dependencies

```
pip install -r model/requirements.txt
```
If you want to use my trained model along with the api provided in **app.py**, download the assets in the latest release and store them in a subdirectory of **/models** called **utils**. 

### Download Dataset 

Download the dataset from the link in the Data and Preprocessing section of this document. 

### Train Model

Run the following:

```
python train.py \
data_dir=[path to directory containing df_train, df_valid files] \
image_dir=[path to directory containing all images] \
output_dir=[path to directory where checkpoints should be saved] \
vocab_path=[path to vocab mapping from integer ids to tokens as a .pkl file] \
--cuda=[include only if training on gpu]
```
