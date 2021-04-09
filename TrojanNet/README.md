# TrojanNet presentation

#### Presenters: Chengkai Su, Faryab Haye, Zuoyi Li

#### Feb 18, 2021


## Paper
TrojanNet Paper:   [HERE](https://arxiv.org/abs/2006.08131)

## Code reading guidance

Please run the demo_zuoyi.ipynb in google colab to re-create our results. This python notebook is based on a fork of the paper's [original repo](https://github.com/trx14/TrojanNet).


## Replication summary [consistency v.s. inconsistency]

Our replication results are consistent with the paper's description. We were able to inject TrojanNet structure into a clean DNN model within seconds, and contaminate the model with minimal to none influence to its performance on the original task sets. The TrojanNet also gets triggered successfully 100% of the time and force the output to a preset value.

