# Presentation: CNN-generated images are surprisingly easy to spot... for now


#### Songlin Liu, Zihan Wang, Bingzhao Shan

#### March 18, 2021

## Paper
Original Paper:   [HERE](https://arxiv.org/abs/1912.11035)


## Code reading guidance

In our replication experiment, we first initialized a binary ResNet-50 classifier with pre-trained weights provided by the authors. Our experiments code are organized as following:

- Setup: model initialization and weight loading [Section1].

- Quantitative Evaluation on BigGAN, CRN, StarGAN, and DeepFake [Section2]

- Fakeness sscore and percentile ranking on BigGAN and StarGAN [Section3]

- Frequency Analysis on CRN, StarGAN, and DeepFake. [Section4]

## Replication summary [consistency v.s. inconsistency]

- In section2, we reproduced some of the quantitative evaluation results on the ForenSynths dataset. We showed the avearge precision, accuracy on real/fake images and the over all accuracy on the following dataset: BigGAN, CRN, StarGAN, StyleGAN2 and deepfake. The AP(average prevision) scores match the number shown in the original paper.

- In section3, we reproduced the fakeness score and fake percentile visualization on BigGAN and StarGAN. As mentioned in the paper, fakeness ranking doesn't have obvious patterns in other datasets other than these two. Among BigGAN's and StarGAN's generated fake images, The overall image quality and fakeness percentile ranking are consistent with the paper in our replication.

- In section4, we reproduced the frequency analysis on CRN, StarGAN and deepfake since these three datasets showed most typical pattern in the original paper. Our replication results are not 100% consistent with the original paper: although our reproduced fake image spectras has similar patterns to the spectras showed in the paper, the contrast is not as big. We think this might be caused by the choice of the kernel size of median blur and the randomness of choosing sample images within the datasets.

