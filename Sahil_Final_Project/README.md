# Final Project - RobEn Extension

#### Sahil Farishta

## Information
This is an extension of the RobEn paper to add defense against adversarial substitutions. The original paper is available [here](https://arxiv.org/abs/2005.01229).

I include the paper and presentation I gave regarding this project, along with the modified roben code and a Google Colab notebook I used to evaluate the trained defense against the TextFooler attack. The RobEn code is primarily the same as the original and is the work of the original authors. The original code is available [here](https://github.com/ejones313/roben). The authors suggest using CodaLab to run their code. I used CodaLab to train the models and used the Google Colab notebook to evaluate their performance against the TextFooler attack (paper available [here](https://arxiv.org/abs/1907.11932)). My contributions are primarily to the preprocess_vocab.py file, where I experimented with using KNNs on the GloVe embeddings to determine synonyms, along with the changes I made to include WordNet based synonyms. 