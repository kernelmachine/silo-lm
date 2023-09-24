raw_file="new-amazon" # new-amazon # cc-news # Wikipedia_(en)
model=kernelmachine/silo-pdsw-1.3b
index_path=/gscratch/zlab/sewon/nplm-inference/out/ours-v1_1.3B_250B_semibalanced/train-0/new-amazon-1024-512-[0K-2000K].index
tokenized_dir="/gscratch/zlab/sewon/nplm-inference/out/neoX/train-0"
dataset=cr
K=1024
KNN_TEMP=1
inter_lambda=0.7

PYTHONPATH=. python scripts/eval.py \
--model ${model}  \
--knn_model ${model} \
--n_sample 3000 \
--raw_file ${raw_file} \
--inter_lambda $inter_lambda \
--index_path $index_path \
--dataset_dir data_eval/benchmark/$dataset \
--k $K \
--dataset_name $dataset \
--batch_size 5 \
--knn_temp ${KNN_TEMP} 
