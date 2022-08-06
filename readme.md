# Deep Learning based Models for Breast Cancer Risk Assessment
This repository contains an implementation of a deep neural network for the risk stratification of breast cancer. 
The model is based on the work published by Wu _et al._ [[1]](#1) for breast cancer classification, and is implemented 
in **Pytorch**. Specifically, pre-trained feature extraction layers are taken and fine-tune for our data and task.
In summary, our method takes a 4-view set of images for each sample and outputs a risk score.
Here, we provide not only the model definitions, but also the pre-processing, training, and testing scripts for reproducibility. 

## Data
The methodology is validated using an assembled case-control study with full-field digital mammography images.
Those mammograms were collected in TAYS with Phillips and General-Electric scanners. The details of the matched cases
and controls are found in the files: `data/input/cases.csv` and `data/input/controls.csv`. Notably,
those files contain the path for each of the views used per sample (`L_CC_path`,`R_CC_path`,`L_MLO_path`,`R_MLO_path`)

## Prerequisites
* python (3.7)
* numpy (1.18.4)
* opencv-python (4.2.0.34)
* future (0.18.2)
* pillow (7.1.2)
* pytorch (1.1.0)
* torchvision (0.6.0)
* cudatoolkit (9.0)
* scipy (1.4.1)
* scikit-image (0.15.0)
* matplotlib (3.1.0)
* imageio (2.8.0)
* pydicom (1.3.0)
* pandas (0.25.0)
* scikit-learn (0.21.2)
* h5py (2.8.0)
* pyprind (2.11.2)
* PyYAML (5.1.1)
* tensorboard
* psutil

## How to run the code
1. Organize case-control study as in files: `data/input/cases.csv`, `data/input/controls.csv`.
In case of using the same data, only modify the root in the paths if necessary.

2. Download [pre-trained models](https://github.com/nyukat/breast_cancer_classifier/tree/master/models) provided by Wu _et al._,
and place them in `breast_cancer_classifier/models/` 

3. Run in terminal `sh run.sh` to execute the whole pipeline. Otherwise, the user can independently run each one of the steps in
the running code.

---
**Optional**

###### Tensorboard

The user can keep tracking the train process by using `tensorboard`. For this purpose, open a terminal,
activate your virtual environment, and run the following commands:

```
cd $REPO_PATH/runs/$EXP_NAME
tensorboard --logdir=./
```
where `$REPO_PATH` is the path to this repository and `$EXP_NAME` is the name of the experiment defined in `config/config.yaml`.
Then, go to your internet browser and type `localhost:6006/`

###### Demo for loading data

We provide a demo to visualize the process of loading data. This can be done by running:

```
python3 demo_load_data.py --show_contours --set train --num_samples 3 --config config/config.yaml
```
Check the arguments defined in the python code for more details.

---

### Pipeline
The pipeline consists of five stages.
1. Crop mammograms
2. Compute optimal crop sizes
3. Calculate optimal centers
4. Train network
5. Get risk scores in validation set

The following variables defined in `run.sh` correspond to the necessary inputs:
* `NUM_PROCESSES`: The number of processes to be used in preprocessing.
* `CONFIG`: Path to config file with settings for training and validation.
* `CASES_DATA`: Path to csv file with the information of cases.
* `CONTROLS_DATA`: Path to csv file with the information of controls.

The other variables define the path to the files that are generated during the execution of the pipeline. 
We do not recommend to change them as it might crash the process.


## References
<a id="1">[1]</a> 
Wu, Nan, et al. 
"Deep neural networks improve radiologists’ performance in breast cancer screening." 
IEEE Transactions on Medical Imaging (2019).