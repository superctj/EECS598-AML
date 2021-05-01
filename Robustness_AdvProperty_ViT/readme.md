# Robustness and Adversarial Properties of Vision Transformers

#### Songlin Liu, Zihan Wang, Bingzhao Shan

#### Apr 30, 2021

### Content

Presentation Report; [HERE](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/EECS598FinalPresentation.pdf)

Final Report:   [HERE](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/Robustness%26AdversarialPropertiesViT.pdf)

### Code reading guidance

1. [VIT_robustness_ImageNet16](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/VIT_robustness_ImageNet16.ipynb) on ImageNet16

- ViT fine-tuning
- ResNet fine-tining
- White-box attacks (FGSM, PGD, C&W)
- White-box attacking transferability
- Adversarial trainning

Note: usage on CIFAR10 can be found here: [ViT_ResNet50_CIFAR10_robustness](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/ViT_ResNet50_CIFAR10_robustness.ipynb)

2. [VIT_detector_resnet](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/VIT_detector_resnet_lr0.00001.ipynb)

- Example of adversarial detector trainer. Usage are similar for other detector settings

Note: [detector_fft_normal_train](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/detector_fft_normal_train.ipynb) is the usage of training a detector under frequency domain.

3. [binary_detectir_eval1](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/binary_detectir_eval1.ipynb)

- Example of adversarial detector evaluation. Usage are similar for other detector settings.

4. [20210429_ensemble_train](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/20210429_shan_ensemble_train.ipynb)

- Example of ensemble trainer and attack.

5. [20210429_ensemble_eval](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/20210429_shan_ensemble_eval.ipynb)

- Example of ensemble attacking evaluation.

6. [Detector training with DCT transform](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/VIT_detector_dct.ipynb)

- Example of Detector training with DCT transform, and both spatial and freuqnecy domain normalization.

7. [Frequency analysis](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/ViT_filter_comparison.ipynb)

- Example of frequency analysis on adversarial examples.

8. [PGD adversarial dataset generation](https://github.com/zuoyigehaobing/EECS598-AML/blob/master/Robustness_AdvProperty_ViT/PGD_dataset_generation.ipynb)

- Example of PGD-5 adversarial dataset generation.

9. [This links to useful weight files](https://drive.google.com/drive/folders/1Oisx57dUR1qMEAuL_m09YKxKJf0pceFJ?usp=sharing)

- This folder contains weights for models we have fine-tuned or trained.

