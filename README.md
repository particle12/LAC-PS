# LAC-PS
This is the code implementation of the paper “LAC-PS: A Light Direction Selection Policy under the Accuracy Constraint for Photometric Stereo”

## dataset
Download the synthetic dataset for training and testing from ReLeaPS: https://drive.google.com/file/d/1hZtjtY8DMOk-sITT_AoZzBs5oZzVdgkk/view
Download the benchmark dataset from https://photometricstereo.github.io/

## checkpoint
For light_direction_selection_network :  https://pan.xunlei.com/s/VOSO8YK4swrbqhxlth6SvSGhA1?pwd=mb5e#
For accuracy_assessment_model：   https://pan.xunlei.com/s/VOSO8iN8xnREGZDVk1ar_-XdA1?pwd=7gnq#


## 1. To train the light_direction_selection_network
```bash
cd light_direction_selection_network
python src/rl_train.py
```

## 2. To test the accuracy_assessment_model
```bash
cd test_accuracy_assessment_model
python src/test_accuracy_assessment_model.py
```

## 3. To test the light_direction_selection_network
```bash
cd light_direction_selection_network
python src/rl_test.py
```

## Acknowledgement:
THis code is partially based on: https://github.com/guanyingc/PS-FCN and https://github.com/jhchan0805/ReLeaPS
