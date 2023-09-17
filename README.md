# Solution of Team sdkxd for MICCAI 2023 Challenges: STS - Tooth Segmentation Task Based on 3D CBCT

Built upon [MIC-DKFZ/nnUNet-1.7.1](https://github.com/MIC-DKFZ/nnUNet/tree/v1.7.1), this repository provides the solution of team sdkxd for [MICCAI 2023 Challenges: STS - Tooth Segmentation Task Based on 3D CBCT](https://tianchi.aliyun.com/competition/entrance/532087/rankingList).

## Environments and Requirements:
Install nnU-Net as below. You should meet the requirements of nnUNet, our method does not need any additional requirements. For more details, please refer to https://github.com/MIC-DKFZ/nnUNet/tree/v1.7.1
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

## 1. Training nnUNet for Labeled Data
### 1.1. Prepare Labeled Data of MICCAI 2023 Challenge
Following nnUNet, give a TaskID (e.g. Task001) to the labeled data and organize them folowing the requirement of nnUNet.

    nnUNet_raw_data_base/nnUNet_raw_data/Task01_Tooth /
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs
    └── labelsTr
### 1.2. Conduct automatic preprocessing using nnUNet.
Here we do use the default setting.
```
nnUNet_plan_and_preprocess -t 1 --verify_dataset_integrity
```
### 1.3. Training nnUNet by 5-fold Cross Validation
```
for FOLD in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=0,1 nnUNet_train_DP 3d_fullres nnUNetTrainerV2_DP 1 $FOLD -gpus 2 -c --npz
done
```

## 2. Generate Pseudo Labels for Unlabeled Data

### 2.1. Generate Pseudo Labels
```
nnUNet_predict -i $INPUTS_FOLDER -o $OUTPUTS_FOLDER -t 2 -m 3d_fullres --save_npz
```

### 2.2. Iteratively Train Models and Generate Pseudo Labels
- Give a new TaskID (e.g. Task002) and organize the Labeled Data and Pseudo Labeled Data as above.
- Conduct automatic preprocessing using nnUNet as above.
  ```
  nnUNet_plan_and_preprocess -t 2 --verify_dataset_integrity
  ```
- Training new nnUNet by all training data
  ```
  for FOLD in 0 1 2 3 4
  do
  CUDA_VISIBLE_DEVICES=0,1 nnUNet_train_DP 3d_fullres nnUNetTrainerV2_DP 2 $FOLD -gpus 2 -c --npz
  done
  ```
- Generate new pseudo labels for unlabeled data.

### 2.3. Filter Low-quality Pseudo Labels
We compare Pseudo Labels in different rounds and filter out the labels with high variants.
```
select_pseudo_label.ipynb
```

### 3. Training nnUNet-att by 5-fold Cross Validation
### 3.1. Copy the following files in this repo to your nnUNet environment.
```
./nnunet/network_architecture/generic_UNet.py
```
### 3.2. Training nnUNet-att by 5-fold Cross Validation

```
for FOLD in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=0,1 nnUNet_train_DP 3d_fullres nnUNetTrainerV2_DP 3 $FOLD -gpus 2 -c --npz
done
```

### 3.3.  Do Efficient Inference with nnUNet-att
We modify the `generic_UNet.py` of nnunet source code for efficiency. Please make sure the code backup is done and then copy the whole repo to your nnunet environment.
```
nnUNet_predict -i $INPUTS_FOLDER -o $OUTPUTS_FOLDER -t 3 -m 3d_fullres --save_npz
```

### 4. Correction of abnormal segmentation points through position correction module

```
python position_correction.py
```