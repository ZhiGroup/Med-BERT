# This script processes Cerner dataset and builds pickled lists including a full list that includes all information for case and controls
# The output data are cPickled, and suitable for training Doctor AI or RETAIN
# Using similar logic of process_mimic Written by Edward Choi (mp2893@gatech.edu) ### updated by LRasmy
# Usage: feed this script with Case file, Control file ,and Case-Control Matching file. and execute like:
# python process_cerner_f_5.py <Case File> <Control File> <Matching File > <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> <train/test/valid>
# Case and contol files should contain pt_id, Diagnosis, Date
# the matching file should contain the case_id, assigned control_id , index_date

# Output files
# <output file>.pts: List of unique Cerner Patient IDs. Created for validation and comparison purposes
# <output file>.labels: List of binary values indicating the label of each patient (either case(1) or control(0)) #LR
# <output file>.dates: List of List of Python datetime objects. The outer List is for each patient. The inner List is for each visit made by each patient
# <output file>.days: List of List of integers representing number of days between consequitive vists. The outer List is for each patient. The inner List is for each visit made by each patient
# <output file>.visits: List of List of List of integer diagnosis codes. The outer List is for each patient. The middle List contains visits made by each patient. The inner List contains the integer diagnosis codes that occurred in each visit
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.
# The above files will also be splitted to train,validation and Test subsets using the Ratio of 75:10:15
# For further resampling and Sorting use the sampling_cerner.py


import sys
from optparse import OptionParser

try:
    import cPickle as pickle
except:
    import pickle

#import pprint

import numpy as np
import random
import pandas as pd
#from pandas import read_table
#from pandas import dataframe
from datetime import datetime as dt
from datetime import timedelta
#import gzip

#import timeit
if __name__ == '__main__':
    
   caseFile= sys.argv[1]
   controlFile= sys.argv[2]
   typeFile= sys.argv[3]
   outFile = sys.argv[4]
   subsetType = sys.argv[5]
   ptlfile = sys.argv[6]
   sample_size= sys.argv[7]
   parser = OptionParser()
   (options, args) = parser.parse_args()

  
   
   
   #_start = timeit.timeit()
   
   debug=False
   np.random.seed(1)
   #visit_list = []
   time_list = []
   dates_list =[]
   label_list = []
   pt_list = []

   
   ### Building the Data starting from the matching file

   print (" Loading cases and controls") 
   
   '''
 #  matchingFile = "samp_hf50_match.csv"
   patients = pd.read_table(matchingFile)
   patients.columns = ["Case_id", "Control_id", "Index_date"]
   '''
   
   ## loading Case
  # caseFile="samp_hf50_case.csv"
   data_case = pd.read_table(caseFile)
   data_case.columns = ["Pt_id", "ICD", "Time"]
   data_case['Label'] = 1
   set_p = pickle.load(open(ptlfile, 'rb'))#,encoding= 'bytes')
   set_p_samp=random.sample(set_p,int(sample_size))
   #cas_sk=data_case['Pt_id']
   #cas_sk=cas_sk.drop_duplicates()
   #cas_sk_samp=cas_sk.sample(n=sample_size)
   #ncases_sub=data_case[data_case['Pt_id'].isin(cas_sk_samp.values.tolist())]
   ncases_sub=data_case[data_case['Pt_id'].isin(set_p_samp)]


    ## loading Control
#   controlFile = "samp_hf50_control.csv"    
   data_control = pd.read_table(controlFile)
   data_control.columns = ["Pt_id", "ICD", "Time"]
   data_control['Label'] = 0
   #ctr_sk=data_control['Pt_id']
   #ctr_sk=ctr_sk.drop_duplicates()
   #ctr_sk_samp=ctr_sk.sample(n=50000)
#   nctrls_sub=data_control[data_control['Pt_id'].isin(ctr_sk_samp.values.tolist())]
   nctrls_sub=data_control[data_control['Pt_id'].isin(set_p_samp)]

   
#   data_l= pd.concat([data_case,data_control])   
   data_l= pd.concat([ncases_sub,nctrls_sub])
   
   ## loading the types
   
   if typeFile=='NA': 
       types={}
   else:
      with open(typeFile, 'rb') as t2:
             types=pickle.load(t2)
        
   #end_time = timeit.timeit()
    ## Mapping cases and controls
  
   #print ("consumed time",(_start -end_time)/1000.0 )
    
   full_list=[]
   index_date = {}
#  The_patient_of = {}
   #visit_list = []
   time_list = []
   dates_list =[]
   label_list = []
   pt_list = []
   dur_list=[]
   #types = {}
   newVisit_list = []
   count=0
   
   for Pt, group in data_l.groupby('Pt_id'):
            data_i_c = []
            #print (group)
            data_dt_c = []
            #for Time, subgroup in group.sort_values(['Time'], ascending=False).groupby('Time', sort=False): ### changing the sort order
            for Time, subgroup in group.sort_values(['Time'], ascending=False).groupby('Time', sort=False): ### changing the sort order
                        data_i_c.append(np.array(subgroup['ICD']).tolist())             
                        data_dt_c.append(dt.strptime(Time, '%Y-%m-%d'))
            #print ('dates', data_dt_c)
            if len(data_i_c) > 0:
                  #cs.append(Pt)
                  #cs.append(group.iloc[0]['Label'])
                  #cs.append(data_dt_c)
                 # creating the duration in days between visits list, first visit marked with 0        
                  v_dur_c=[]
            if len(data_dt_c)<=1:
                     v_dur_c=[0]
            else:
                     for jx in range (len(data_dt_c)):
                         if jx==0:
                             v_dur_c.append(jx)
                         else:
                             #xx = ((dt.strptime(data_dt_c[jx-1], '%d-%b-%y'))-(dt.strptime(data_dt_c[jx], '%d-%b-%y'))).days
                             xx = (data_dt_c[jx-1]- data_dt_c[jx]).days                             
                             v_dur_c.append(xx)
                                    
                  #cs.append(v_dur_c)
                  #cs.append(data_i_c)

                ### Diagnosis recoding
#                 print 'Converting cerner codes to a sequential integer code, and creating the types dictionary'
            #print ('dur', v_dur_c)                  
            newPatient_c = []
            for visit in data_i_c:
                      #print visit
                      newVisit_c = []
                      for code in visit:
                        #if code.startswith('D'):
                        				if code in types: newVisit_c.append(types[code])
                        				#else:                             
                        				#	  types[code] = len(types)+1
                        				#	  newVisit_c.append(types[code])
                      #print types
                      #print newVisit_c 
                      if len(newVisit_c) >0: newPatient_c.append(newVisit_c)
                  #cs.append(newPatient_c)

 #                 print cs
                                                            
            if len(newPatient_c) > 0: ## only save non-empty entries
                  #visit_list.append(data_i_c)
                  newVisit_list.append(newPatient_c)
                  #dates_list.append(data_dt_c)
                  dur_list.append(v_dur_c)
                  label_list.append(group.iloc[0]['Label'])
                  pt_list.append(Pt)
                  #cs_ct.append(cs)
 
            count=count+1
            if count % 1000 == 0: print ('processed %d pts' % count)

   ptee_list = []
   
#   print visit_list 
#   print newVisit_list 
#   print dates_list
#   print dur_list
#   print label_list
#   print pt_list
#   print full_list
#   print types

   
    ### Creating the full pickled lists
   #outFile='LRTest3'

   pickle.dump(label_list, open(outFile+'.labels.'+subsetType, 'wb'), -1)
   pickle.dump(newVisit_list, open(outFile+'.visits.'+subsetType, 'wb'), -1)
   pickle.dump(types, open(outFile+'.types.'+subsetType, 'wb'), -1)
   pickle.dump(pt_list, open(outFile+'.pts.'+subsetType, 'wb'), -1)
   #pickle.dump(dates_list, open(outFile+'.dates.'+subsetType, 'wb'), -1)
   pickle.dump(dur_list, open(outFile+'.days.'+subsetType, 'wb'), -1)
   #pickle.dump(visit_list, open(outFile+'.visitsCC', 'wb'), -1)
   #pickle.dump(full_list, open(outFile+'.CsCt.'+subsetType, 'wb'), -1)

   ### Create the combined list for the Pytorch RNN
   fset=[]
   print ('Reparsing')
   for pt_idx in range(len(pt_list)):
                pt_sk= pt_list[pt_idx]
                pt_lbl= label_list[pt_idx]
                pt_vis= newVisit_list[pt_idx]
                pt_td= dur_list[pt_idx]
                d_gr=[]
                n_seq=[]
                d_a_v=[]
                for v in range(len(pt_vis)):
                        nv=[]
                        nv.append([pt_td[v]])
                        nv.append(pt_vis[v])                   
                        n_seq.append(nv)
                n_pt= [pt_sk,pt_lbl,n_seq]
                fset.append(n_pt)              
    
   ### split the full combined set to the same as individual files

   ctrfilename=outFile+'.combined.'+subsetType
   pickle.dump(fset, open(ctrfilename, 'wb'), -1)


