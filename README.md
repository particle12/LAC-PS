# LAC-PS
This is the code implementation of the paper “LAC-PS: A Light Direction Selection Policy under the Accuracy Constraint for Photometric Stereo”

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