# AugTrojanNet final project

EECS 598-07 AML Luya Gao

This folder contains an adjusted Trojan. The majority of the code is copied from the TrojanNet repo:https://github.com/trx14/TrojanNet (Authors: Ruixiang Tang, Mengnan Du, Ninghao Liu, Fan Yang, Xia Hu) with the following adjustment to facilitate some augmentations on the original network:

- AugTrojanNet (aug_trojannet.py): base structure is the same as TrojanNet (https://github.com/trx14/TrojanNet/blob/master/code/TrojanNet/trojannet.py) except for adding the logic of spreading trigger pixels across image and regathering them
- Detection code slightly adjusted for Tensorflow 2.0 (neural_cleanese)
- a jupyter notebook containing the process of activation map generation (neural_cleanese/activation_map_generation.ipynb)
- Adding AugTrojanNet to neural_cleanese/gtsrb_visualize_example.py to enable NeuralCleanse detection with AugTrojanNet
- result/results_augtrojan: reverse engineered trigger patterns of AugTrojanNet