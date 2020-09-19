# Med-BERT
This repository provides the code for pre-training and fine-tuning Med-BERT, a contextualized embedding model that delivers a meaningful performance boost for real-world disease-prediction problems as compared to state-of-the-art models.

### Overview
Med-Bert adapts bidirectional encoder representations from transformers (BERT) framework and pre-trains contextualized embeddings for diagnosis codes mainly in ICD-9 and ICD-10 format using structured data from an EHR dataset containing 28,490,650 patients. 
 ![Med-BERT_Structure](Med-BERT_Structure.png)
Please refer to our paper [Med-BERT: pre-trained contextualized embeddings on large-scale structured electronic health records for disease prediction](https://arxiv.org/abs/2005.12833) for more details.

  
## Reproduce Med-BERT
#### Pretraining

To reproduce the steps necessary for pre-training Med-BERT

    python data_preprocess.py 
    python create_ehr_pretrain_data.py
    python run_EHRpretraining.py   ##(Tensorflow Based)

#### Fine-tuning Tutorial

To see an example of how to use Med-BERT for a specific disease prediction task, you can follow the [Med-BERT DHF prediction notebook]

Kindly note that you need to use the following code for preparing the fine-tunning data:
    python create_ehr_pretrain_FTdata.py


### Dependencies
    Python: 3.7+
    Pytorch 1.5.0
    Tensorflow 1.13.1+
    Pandas
    Pickle
    tqdm
    pytorch-transformers
    Google BERT
    

### Results
 ![Med-BERT Results](Med-BERT%20results.jpg) 
<B>Prediction results for the evaluation sets by training on different sizes of data on DHF-Cerner (top), PaCa-Cerner (middle), and PaCa-Truven (bottom). The shadows indicate the standard deviations. Please refer to our paper for more details.
 
### Contact

Please post a Github issue if you have any questions.

### Citation

Please acknowledge the following work in papers or derivative software:

Rasmy, Laila, Yang Xiang, Ziqian Xie, Cui Tao, and Degui Zhi. "Med-BERT: pre-trained contextualized embeddings on large-scale structured electronic health records for disease prediction." arXiv preprint arXiv:2005.12833 (2020).



