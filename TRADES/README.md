# TRADES Presentation

EECS 598-07 AML Luya Gao

This folder contains the robustness evaluation results of [TRADES](https://github.com/yaodongyu/TRADES) created by the authors of the paper [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/pdf/1901.08573.pdf) Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan. 

## Results
The original evalution code only counts the number of instances that are misclassified. I added code that count the total number of instances and print out the percentage that is misclassified. As the results in the output file shows, the natural error percentage is 0.1508, and the robustness error percentage is 0.4289. This is equivalent to having a natural accuracy of 84.92% and a robustness accuracy of 57.11%, which is close to the results reported by the authors in table 5 of their paper.