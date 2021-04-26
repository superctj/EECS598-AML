# Bayesian Relevance
Ananth Chillarige, Kevin Lee and Shrijesh Siwakoti
## 
The [notebook](advml_bayesian.ipynb) contains steps to download, setup and run the BayesianRelevance Network. The experiment was run with FGSM attacks on the Determinant Neural Network and the Bayesian Neural Network trained on the MNIST Dataset. LRP was used to calculate the robustness of the Bayesian Network against FGSM attack. The weights, graphs and saved models from the experiments are in the [experiments](experiments) folder.

The original paper on BayesianRelevance is available here: [Resilience of Bayesian Layer-Wise Explanations under Adversarial Attacks](https://arxiv.org/abs/2102.11010#:~:text=Resilience%20of%20Bayesian%20Layer%2DWise%20Explanations%20under%20Adversarial%20Attacks,-Ginevra%20Carbone%2C%20Guido&text=Our%20results%20not%20only%20confirm,assessments%20of%20Neural%20Network%20predictions.)

The official source code for the paper is available here: [ginevracoal/BayesianRelevance](https://github.com/ginevracoal/BayesianRelevance)

The official source code was forked to fix depricated packages and setup experiments and is available here: [ssiwakot/BayesianRelevance](https://github.com/ssiwakot/BayesianRelevance)

DeepRobust Library was forked and updated to fix compatability issues and can be found here: [ssiwakot/DeepRobust](https://github.com/ssiwakot/DeepRobust)
