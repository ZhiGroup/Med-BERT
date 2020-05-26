### This file is to convert previous data preprocessed for Baeline RNN models to be Bert compatable
### --- need to add the shape from to -- later

import pickle
import argparse

parser = argparse.ArgumentParser(description='Convert data format to be BERT compatible')
parser.add_argument('-splitted', type=bool, default=True, help='indicator of whether the file is already splitted to train, Test and validation or you just need to reformat a single_file. [default: True]')
parser.add_argument('-filename', type = str, default = '../pdata/' , help='the prefix for your input file it is splitted , or if it is a single file gove the full file path/name')
args = parser.parse_args()
print ( "parameters :" ,args.splitted,args.filename)
if args.splitted:
    print("splitted, will load Train , Test , and valid subsets")
    train_sl= pickle.load(open(args.filename+'.train', 'rb'), encoding='bytes')
    test_sl= pickle.load(open(args.filename+'.test', 'rb'), encoding='bytes')
    valid_sl= pickle.load(open(args.filename+'.valid', 'rb'), encoding='bytes')
    
    datasets=[train_sl,test_sl,valid_sl]
    file_names=[args.filename+'_Bert.train',args.filename+'_Bert.test',args.filename+'_Bert.valid']
else:
    print("non splitted, will load single file")
    datasets=[pickle.load(open(args.filename, 'rb'), encoding='bytes')]
    file_names=[args.filename+'_Bert']

print('Repasring')
for d,set_sl in enumerate(datasets):
    new_set_sl=[]
    for pt in set_sl:
        in_seq=[]
        in_vseg=[]
        in_tdseg=[]
        n_pt=[]
        v=0
        n_pt.append(pt[0])
        for visit in pt[-1]:
            i=visit[-1]
            #t=visit[0]
            v=v+1
            in_seq.extend(i)
            in_vseg.extend([v]*len(i))
            in_tdseg.extend(visit[0]*len(i))
  
        in_pos=list(range(1,len(in_seq)+1))
        n_pt.append(in_seq)
        n_pt.append(in_vseg)
        #n_pt.append(in_tdseg)
        #n_pt.append(in_pos)
        ###print (n_pt)
        new_set_sl.append(n_pt)
    pickle.dump(new_set_sl, open(file_names[d], 'wb'), -1)
    print(file_names[d],"created with ",len (new_set_sl),' patients')
