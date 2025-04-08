# DeGen: Decision boundary generation through latent style manipulation (MICCAI 25)
This repository contains the source code of the paper *DeGen: Decision boundary generation through latent style manipulation*

## Preparation
### Installing the Environment
To install the environment simply run the following command:
```conda env create -f conda_env.yaml ```
### Preparing Datasets
In theory, the proposed method works on any binary classification dataset. However, as many datasets' structures are constructed differently, we chose to construct datasets and dataloaders using a simple CSV file that contains the path to the image and image label. To run the model, prepare the train/val/test sets' csv files such that there are two columns: ```img_path``` and ```img_label``` with an optional ```img_id``` column. Since the core of this repository is based on [UVCGAN2](https://github.com/LS4GAN/uvcgan2) and there are options for paired and unpaired training, we need to separate each train/val/test sets into domain A and B. At the end, the dataset csvs should be saved in the following format:
```
├── dataframes
│   ├── train_A.csv
│   ├── train_B.csv
│   ├── val_A.csv
│   ├── val_B.csv
│   ├── test_A.csv
│   ├── test_B.csv
```
## Translation Model
### Training
The first phase of our method is training the image translation model. To do so, simply run the command:
```python3 train_decisionViT.py --dataset DATASET```
For other variations of the model and data configurations use please see set the appropriate configurations using commandline arguments. Some common arguments are given below:
```
--img_ch (int) [1, 3] # input image channels
--lambda-fwd (float) # class prediction weight
```
### Evaluation
To evaluate the quality of the generated images, we can use the functions in ./classification/eval_quality.py


### Generation
#### Interpolation 

## Classification Model
### Training
### Evaluation

## Acknowledgements
The image translation model we implemented our method on is ```UVCGAN2```
* [UVCGAN v2: An Improved Cycle-Consistent GAN for Unpaired Image-to-Image Translation](https://github.com/LS4GAN/uvcgan2)
