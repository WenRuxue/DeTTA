# DeY-Net
This is the code for:

> [**From Denoising Training to Test-Time Adaptation: Enhancing Domain Generalization for Medical Image Segmentation**](https://arxiv.org/abs/2310.20271)<br>

## Requirements
- Python 3.6 or 3.7 are supported.
- Pytorch 1.4.0 + is recommended.
- This code is tested with CUDA 11.1 toolkit and CuDNN 8.0.0.
- Please check the python package requirement from [`requirements.txt`](requirements.txt), and install using
```
pip install -r requirements.txt
```
## Data Preparation
### Datasets
We use two public datasets: CHAOS dataset and LITS dataset.
We process them through the following steps:
1. Run Data_process/read_nii.py, slicing the .nii files to .png files.
2. Run Data_process/list.py, generating the training and validating set in txt.
3. For .dicom files, it is the same as read_nii.py, except for the part of reading the file.
```
    dicom_file = sitk.ReadImage(os.path.join(root, file))
    pixel_array = sitk.GetArrayFromImage(dicom_file)
```

### File Organization

CHAOS dataset as source domain
``` 
├── [Your Path]
    ├── CHAOS
        ├── label_data
            ├── image
                ├── 000_000.png, 000_001.png, xxx
            └── label
            └── train_list.txt
            └── val_list.txt
        └── unlabel_data
            ├── image
            └── unlabel_list.txt
```

## Training and Testing

### Pretraining
```
python pretrain.py \
--seed [random seed]
--data [path to the source domain dataset]
--save_path [logs and model path]
```
### Training
```
python train_DeYNet.py \
--seed [random seed]
--data [path to the source domain dataset]
--save_path [logs and model path]
--pretrain_path [pretrain model path]
--pretrain_part [pretrain encoder/decoder_seg/decoder_de]
```
### Test-Time Adaptation 
```
python predict_DeYNet.py \
--seed [random seed]
--data_dir [path to the target dataset]
--mode_list [target datasets: ['test','LITS']]
--step [optimizing step]
--average [adapts to noise-corrupted input or not]
```
## Comparison Methods
### U-Net
```
python train_UNet.py
python predict_UNet.py
```
### TTT
```
python train_TTT.py
python predict_TTT.py
```
### Tent
```
python train_UNet.py
python Tent_for_UNet.py
```
### RN+CR
```
python train_UNet.py
python RN+CR_for_UNet.py
```