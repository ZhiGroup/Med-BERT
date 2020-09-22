## Pretraining Tutorial

The main purpose of this page is to explain the pretraing process step by step. We mainly used the pretraining code from https://github.com/google-research/bert and  followed their pretraining instructions. The main differences are in the used data, as we mainly trained on structured diagnosis data and therefore we will focus here on the data preparation steps.

### Data Extraction Criteria


### Extracted data format
 

You can find an example for the construction of the data_file under [Example data](Pretraining%20Code/Data%20Pre-processing%20Code/Example%20data) as well as images showing the construction of preprocessed data and the BERT features.

![Med-BERT extracted data format](Data%20Pre-processing%20Code/Example%20data/Data_File.JPG) 
### Preprocessing the extracted data to pickled lists
    python preprocess_pretrain_data.py <data_File> <vocab/NA> <output_Prefix> <subset_size/0forAll>

![Med-BERT preprocessed data](Data%20Pre-processing%20Code/Example%20data/preprocessed%20data.JPG) 

### Converting pickled lists to TF features
    python create_BERTpretrain_EHRfeatures.py --input_file=<output_Prefix.bencs.train> --output_file='output_file' --vocab_file=<output_Prefix.types>--max_predictions_per_seq=1 --max_seq_length=64

![Med-BERT Features format](Data%20Pre-processing%20Code/Example%20data/BERT%20Feature%20structure.JPG) 

### MedBERT Pretraining
    python run_EHRpretraining.py --input_file='output_file' --output_dir=<path_to_outputfolder> --do_train=True --do_eval=True --bert_config_file=config.json --train_batch_size=32 --max_seq_length=512 --max_predictions_per_seq=1 --num_train_steps=4500000   --num_warmup_steps=10000 --learning_rate=5e-5



### Converting TF model to pytorch model
    python 
You can find an example for the construction of the data_file under [Example data](Pretraining%20Code/Data%20Pre-processing%20Code/Example%20data) as well as images showing the construction of preprocessed data and the BERT features.

