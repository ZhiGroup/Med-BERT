# lrasmy @ Zhilab 2019/08/19 a modification of the original BERT pretrain data preparation  
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/LOS masked_lm TF examples for BERT."""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import collections
import random
#import tokenization
import tensorflow as tf
import pickle
import sparkpickle as pkl
import glob
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

## this the visits file
flags.DEFINE_string("input_file", None, "Input training data.")
flags.DEFINE_string("pt_file", None, "pts file to limit the new cohort to")

## that will be the final model input file
flags.DEFINE_string( "output_file", None, "Output TF example file (or comma-separated list of files).")


flags.DEFINE_integer("max_seq_length", 32, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 2, "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_float("masked_lm_prob", 0.35, "Masked LM probability.")

#### this is the types file
flags.DEFINE_string("vocab_file", None, "The vocabulary file / types file that the BERT model was trained on.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

#flags.DEFINE_float("short_seq_prob", 0.7,"Probability of creating sequences which are shorter than the maximum length.") ## commented as not  used


def write_EHRinstance_to_example_files(seqs,max_seq_length, max_predictions_per_seq,masked_lm_prob,vocab, output_files,rng):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0
  total_written = 0
  min_seq_l=max_seq_length ###initialization to be overwritten in the code below
  max_seq_l=0
  for (seq_index, seq) in enumerate(seqs):
    if len(seq[3])> max_seq_length: ### here I excluded patients with longer than 512 code (I keep for comparison purpose, I'd change that later to make input_id = seq[-3][-max_seq_length:])
      continue
    #if seq[3][0]<=0: ### this was to exclude patients who died, not include that element in this version of data
    #  continue 
    if len(seq[3])< min_seq_l:
      min_seq_l=len(seq[-3])
      print(min_seq_l)
    if len(seq[3])> max_seq_l:
      max_seq_l=len(seq[-3])
      print(max_seq_l)
    input_seq = seq[3]
    input_mask = [1] * len(input_seq)
    segment_ids = seq[4].copy()
    
    ##Masking of input
    (input_ids, masked_lm_positions, masked_lm_ids)=create_masked_EHR_predictions(input_seq, masked_lm_prob,max_predictions_per_seq, vocab, rng)  
    assert len(input_ids) <= max_seq_length
    
    
#####here I'm done 
    
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

#### Here I need to check the masking code

    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

##### In this version I replace the QA thing with the binary label for long LOS (>7 days) 

    if max(seq[1])>7:
      next_sentence_label=1 #### here it is time between 2 visits
    else: next_sentence_label=0
    
#### That is the output I need
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if seq_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: " , seq)

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
      #print (features)

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)




MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"]) ## mainly used to sort the data below

def create_masked_EHR_predictions(input_seq, masked_lm_prob,max_predictions_per_seq, vocab, rng):
  """Creates the predictions for the masked LM objective."""
  
  cand_indexes=list(range(len(input_seq)))### LR 4/29 remove[1:]
  rng.shuffle(cand_indexes)
  output_tokens = input_seq[:] ### added slicing to inhibit original list update

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(input_seq) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict: ### LR 4/29 remove >=
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token=0 
    
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      #masked_token = "[MASK]"
      masked_token=0
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        #masked_token = tokens[index]
        masked_token=input_seq[index] ### LR 4/29 added +1
      # 10% of the time, replace with random word
      else:
        #masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
        masked_token=rng.randint(1,292600) #### hardcoding it for now ie replace max(vocab.keys()) with 292600
          
    output_tokens[index] = masked_token ### LR 4/29 added +1

    masked_lms.append(MaskedLmInstance(index=index, label=input_seq[index])) ### Currently keeping the original code but I need to optimize that later from here till end of function

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)
  return (output_tokens, masked_lm_positions, masked_lm_labels)

############ Masking Done

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def main(_):
  #tf.logging.set_verbosity(tf.logging.INFO)

  vocab = pickle.load(open(FLAGS.vocab_file, 'rb'), encoding='bytes')
  
#### LR amended July 2021 to use spark distributed pickle files

  if FLAGS.pt_file!= None:
    pts_include=pickle.load(open(FLAGS.pt_file, 'rb'))
  
  all_files1 = glob.glob(FLAGS.input_file+"/p*")

  train_data=[]
  for file in all_files1:
      tp = pkl.load(open(file,'r+b'))
      
      if FLAGS.pt_file!= None:
        pt_list=[]
        for x in tp:
            pt_list.append(x[0])
        pt_indices = np.intersect1d(pt_list,pts_include,assume_unique=True, return_indices=True)[1]
        subset_p = [tp[i] for i in pt_indices]
        train_data.extend(subset_p)
      else:
        train_data.extend(tp)
#### comment the old version
#  train_data=[]
#  for file in all_files1:
#      with open(file,'r+b') as f:
#        try:               
#            while True:
#                tp=pkl.load(f, encoding='bytes')
#                train_data.extend(tp)
#        except EOFError:
#                pass

  rng = random.Random(FLAGS.random_seed)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)


  write_EHRinstance_to_example_files(train_data,FLAGS.max_seq_length, FLAGS.max_predictions_per_seq,FLAGS.masked_lm_prob,vocab,output_files,rng)

if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()