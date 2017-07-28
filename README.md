# CBIR
Image retrival by caffe

How the use this project files
step-1
Finetune the model base on your image data
step-2
compute_fea_for_cbir.py
Compute the image feature by the finetuned model, and save the to pkl files
step-3
dump_data_to_pkl.py
Collect the feature pkl files to dic format: id_fc7_fc8
step-4
dump_lsh_to_pkl.py
Dump the fc8 features by lsh
step-5
app_cbir.py
Upload a imagefile and search the similar images
