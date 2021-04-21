# RewritingSampler Presentation

EECS 598-07 AML Luya Gao

This folder contains an attack that utilizes the [fibber library](https://github.com/DAI-Lab/fibber) created by the authors of the paper [Rewriting Meaningful Sentences via Conditional BERT Sampling — and an application on fooling text classifiers](https://arxiv.org/pdf/2010.11869.pdf) Lei Xu, Ivan Ramirez, and Kalyan Veeramachaneni. 

## Prerequisites
In order to run the code in this folder, first install python 3.6, 3.7 or 3.8, tensorflow>=2.0.0, and pytorch>=1.5.0 (prerequisites listed in the fibber repo). Then install fibber with pip or from source according to the instructions given in the [repo] (https://github.com/DAI-Lab/fibber). 

If it is the first time that you run RewritingSampler_attack.py, please call Fibber's download_all() to download all the datasets and resources needed (in a similar fashion as the download_resources() function in RewritingSampler_attack.py). Then change the dataset_name in the main() function to the dataset that you would like to attack (available datasets include "ag", "ag_no_title", "mr", "imdb", "yelp", "snli", "mnli", "mnli_mis" as outlined in [Fibber's datasets/_init_.py](https://github.com/DAI-Lab/fibber/blob/master/fibber/datasets/__init__.py)), and run python3 -m fibber.datasets.download_datasets in your terminal.