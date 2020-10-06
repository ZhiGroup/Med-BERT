## Pretraining Tutorial

The main purpose of this page is to explain the pretraining process step by step. We mainly used the model pretraining and optimization code from https://github.com/google-research/bert. As we mainly trained Med-BERT on structured diagnosis data our data preparation and preprocessing pipeline is slightly different than processing free text like original BERT or other variants like Clinical-BERT

### Data Extraction

Based on our pretraining experiments formulation, we extracted all patients who have more than 3 diagnosis codes in their medical records and verified the quality of their records. As our first phase was focusing on training Med-BERT on ICD9/10 diagnosis codes, we extract the diagnosis information and linked that to the encounter (visit) admission and discharge dates. We used these dates to calculate the LOS per encounter as well as the time between 2 consecutive encounters.  
As Cerner HealthFacts doesn't include the exact dates of diagnosis, we decided to use data elements like Present on Admit (POA) flags along with diagnosis priority and if the diagnosis were recorded in EHR or only added on billing (using Third-party system), to sort our diagnosis codes within the encounter.

You can find an example for the extracted data format here [Example data](Data%20Pre-processing%20Code/Example%20data/data_file.tsv) 

![Med-BERT extracted data format](Data%20Pre-processing%20Code/Example%20data/Data_File.JPG) 

### Preprocessing the extracted data to pickled lists

Similar to our previous work at https://github.com/ZhiGroup/pytorch_ehr , we preprocess and store our data as a pickled list for computational efficiency. 

You can run the following to convert the input data format shown above to a pickled list that can be used in later steps

    python preprocess_pretrain_data.py <data_File> <vocab/NA> <output_Prefix> <subset_size>

data_File: is the path for the extracted tab-delimited file
vocab: is the path of a vocabulary (types) file which includes a dictionary that maps different ICD9/10 codes to integer values.
if you have a pre-existing vocab file, you can use the path for such a file or you can use NA to create a new one.
output_Prefix: will be the prefix assigned to output files (more details can be found in [preprocess_pretrain_data.py](Data%20Pre-processing%20Code/preprocess_pretrain_data.py) header
<subset_size>: if you need to only preprocess a subset of the data, specify the subset size here (that should be the number of patients to be included), if you set it to 0 it will include all data.

The output will be a list of patients and each patient is a pickled list like:

<img src="Data%20Pre-processing%20Code/Example%20data/preprocessed%20data.JPG" alt_test="Med-BERT preprocessed data" width="90%" height="70%">

The data will be in 3 splits with a ratio of 7:1:2 for training, validation, and test sets

### Converting pickled lists to TF features

As the BERT Tensorflow model mainly consume TF features (similar to an ordered dictionary), you will need to run the following statement to convert the above created pickled list to TF features

    python create_BERTpretrain_EHRfeatures.py --input_file=<output_Prefix.bencs.train> --output_file='output_file' --vocab_file=<output_Prefix.types>--max_predictions_per_seq=1 --max_seq_length=64
    

Each patient features will look like

<img src="Data%20Pre-processing%20Code/Example%20data/BERT%20Feature%20structure.JPG" width="60%" height="60%">

Note: You will need to run the above statement to each of the training, validation, and test subsets

### MedBERT Pretraining

For the MedBERT Pretraining you can run a command like:

    python run_EHRpretraining.py --input_file='output_file' --output_dir=<path_to_outputfolder> --do_train=True --do_eval=True --bert_config_file=config.json --train_batch_size=32 --max_seq_length=512 --max_predictions_per_seq=1 --num_train_steps=4500000   --num_warmup_steps=10000 --learning_rate=5e-5

You can replace [run_EHRpretraining.py](run_EHRpretraining.py) with [run_EHRpretraining_QA2Seq.py](run_EHRpretraining_QA2Seq.py) to ensure that your pretraining classification task is using the whole sequence rather than the first token

you can monitor the pretraining evaluation using Tensorboard  using command like:
    
    tensorboard --logdir <path_to_outputfolder>

### Converting TF model to pytorch model

As our fine-tuning code is mainly based on the Pytorch framework, we used the huggingface transformers package tools/APIs to convert the TF pre-trained model to Pytorch based models.

You can use either transformers-cli or pytorch_transformers 

    transformers-cli convert --model_type bert --tf_checkpoint <path_to_outputfolder/checkpoint> --config config.json --pytorch_dump_output <path_to_Output_model.bin>
    
    pytorch_transformers bert <path_to_outputfolder/checkpoint> <path_to_configfile.json> <path_to_Output_model.bin>

As we mainly use the full sequence as the input for our fine-tuning, you can ignore warnings regarding [CLS]-related parameters.
For pytorch converted files, you might need to make sure that the model and config files are within the same folder


