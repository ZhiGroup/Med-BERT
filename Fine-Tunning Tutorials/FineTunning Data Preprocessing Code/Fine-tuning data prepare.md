1. Map your features to the MBv2 token ids, you can do that directly after data extraction, either using sql or python
2. Create a case and control files
3. Preprocess the data as:
   
python3.7 preprocess_FT_combined_mbv2.py ../../newFT_Data/mbv2_FT_data/lr_DHF_tmp5_case_mbv2.txt  ../../newFT_Data/mbv2_FT_data/lr_DHF_tmp5_ctrl_mbv2.txt ../../newFT_Data/mbv2_FT_data/lr_DHF_tmp5_mbv2_v1_60k NA 30000

python3.7 ConvertComb_toBertFT.py -filename ../newFT_Data/mbv2_FT_data/lr_DHF_tmp5_mbv2_v1_60k.combined
