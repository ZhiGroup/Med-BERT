## Pretraining Tutorial

The main purpose of this page is to explain the pretraing process step by step. We mainly used the pretraining code from https://github.com/google-research/bert and  followed their pretraining instructions. The main differences are in the used data, as we mainly trained on structured diagnosis data and therefore we will focus here on the data preparation steps.

### Data Extraction Criteria


### Extracted data format


### Preprocessing the extracted data to pickled lists


### Converting pickled lists to TF features


### MedBERT Pretraining


### Converting TF model to pytorch model
