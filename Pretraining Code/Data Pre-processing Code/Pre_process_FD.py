# This script processes Cerner dataset and builds pickled lists including a full list that includes all patient encounters information 
# The output data are cPickled, and suitable for training of BERT_EHR models 
# Usage: feed this script with patient targets file that include patient_id, encounter_id and other relevant labels field ( here we use mortality, LOS and time to readmit) 
# and the main data fields like diagnosis, procedures, medication, ...etc and if you decide to use a predefined vocab file (tokenization/ types dict)
# additionally you can specify sample size , splitting to train,valid,and test sets and the output file path
# So you can run as follow
# python Pre_process_FD.py <target_file> <data_File>  <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> <train/test/valid>
# Case and contol files should contain pt_id, Diagnosis, Date
# Output files:
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.
# <output file>.ptencs: List of pts_encs_data
# <output file>.encs: slimmer list that only include tokenized encounter data and labels
# The above files will also be splitted to train,validation and Test subsets using the Ratio of 70:10:20


import sys
from optparse import OptionParser

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
import random
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta

#import timeit
if __name__ == '__main__':
    
   targetFile= sys.argv[1]
   diagFile= sys.argv[2]
   typeFile= sys.argv[3]
   outFile = sys.argv[4]
   p_samplesize = int(sys.argv[5]) ### replace with argparse later

   parser = OptionParser()
   (options, args) = parser.parse_args()
 
   
   #_start = timeit.timeit()
   
   debug=False
   #np.random.seed(1)

   #### Data Loading

   print (" Loading target and data files" )   
   data_target= pd.read_csv(targetFile, sep='\t')
   data_diag= pd.read_csv(diagFile, sep='\t')   

   if typeFile=='NA': 
       types={}
   else:
      with open(typeFile, 'rb') as t2:
             types=pickle.load(t2)


   #### Sampling
   
   if p_samplesize>0:
    print ('Sampling')
    ptsk_list=data_target['patient_sk'].drop_duplicates()
    pt_list_samp=ptsk_list.sample(n=p_samplesize)
    n_data_target= data_target[data_target["patient_sk"].isin(pt_list_samp.values.tolist())]  
   else:
    n_data_target=data_target
    
   n_data_target.admit_dt_tm.fillna(n_data_target.discharge_dt_tm, inplace=True)
   
   ##### Data pre-processing
   print ('Start Data Preprocessing !!!')
   count=0
   pts_ls=[]
   all_encs_d={}
   for Pt, group in n_data_target.groupby('patient_sk'):
    group = group.sort_values(['discharge_dt_tm'], ascending=True)
    pt_ls=[Pt , len(group[group['pt_type']=='I']),len(group[group['pt_type']=='E'])]
    pt_enc_ids = np.array(group['encounter_id']).tolist()
    pt_mort = np.array(group['mortality']).tolist()
    pt_los = np.array(group['los']).tolist()
    pt_addt = np.array(group['admit_dt_tm']).tolist()
    pt_discdt = np.array(group['discharge_dt_tm']).tolist()
    
    if len(pt_enc_ids)> 0:
            pt_encs=[]
            for ei,eid in enumerate(pt_enc_ids):
                if ei==0:
                    enc_td=0
                else:
                    enc_td =((dt.strptime(pt_addt[ei], '%Y-%m-%d %H:%M:%S'))-(dt.strptime(pt_discdt[ei-1], '%Y-%m-%d %H:%M:%S'))).days
               
                diag_l=np.array(data_diag[data_diag['encounter_id']==eid].sort_values(['poa','third_party_ind','diagnosis_priority'], ascending=True)['diagnosis_id'].drop_duplicates()).tolist()
                if len(diag_l)> 0:
                    diag_lm=[]
                    for code in diag_l:
                        if code in types: diag_lm.append(types[code])
                        else:                             
                            types[code] = len(types)+1
                            diag_lm.append(types[code])

                    enc_l=[eid,pt_mort[ei],pt_los[ei],enc_td,diag_l,diag_lm]
                    pt_encs.append(enc_l)
                    all_encs_d[eid]= [pt_mort[ei],pt_los[ei],enc_td,diag_lm]
                                    
                    
            if len(pt_encs)>0:
                pt_ls.append(pt_encs)
                pts_ls.append(pt_ls)
                count=count+1
            
            if count % 1000 == 0: print ('processed %d pts' % count)
            
  
   ### Random split to train ,test and validation sets
   print ("Splitting")
   dataSize = len(pts_ls)
   np.random.seed(0)
   ind = np.random.permutation(dataSize)
   nTest = int(0.2 * dataSize)
   nValid = int(0.1 * dataSize)
   test_indices = ind[:nTest]
   valid_indices = ind[nTest:nTest+nValid]
   train_indices = ind[nTest+nValid:]

   for subset in ['train','valid','test']:
       if subset =='train':
            indices = train_indices
       elif subset =='valid':
            indices = valid_indices
       elif subset =='test':
            indices = test_indices
       else: 
            print ('error')
            break
       subset_ptencs = [pts_ls[i] for i in indices]
       subset_encs=[]
       for xs in subset_ptencs:
           for i in xs[-1]:
                l=[i[0]]
                l.extend(all_encs_d[i[0]])
                subset_encs.append(l)
       ptencsfile = outFile +'.ptencs.'+subset
       encsfile = outFile +'.encs.'+subset
       pickle.dump(subset_ptencs, open(ptencsfile, 'wb'), -1)
       pickle.dump(subset_encs, open(encsfile, 'wb'), -1)

   pickle.dump(types, open(outFile+'.types', 'wb'), -1)

  

