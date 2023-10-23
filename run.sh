for dataset in cr agn amazon dbpedia hyp mr rotten_tomatoes rte sst2 yelp
do
    raw_file="Wikipedia_(en)" # new-amazon # cc-news # Wikipedia_(en)
    model=kernelmachine/silo-pdsw-1.3b
    K=1024
    KNN_TEMP=1
    inter_lambda=0.7

    PYTHONPATH=. python scripts/eval.py \
        --model ${model}  \
        --knn_model ${model} \
        --n_sample 10 \
        --raw_file ${raw_file} \
        --inter_lambda $inter_lambda \
        --dataset_dir data_eval/benchmark/$dataset \
        --k $K \
        --dataset_name $dataset \
        --batch_size 5 \
        --knn_temp ${KNN_TEMP} \
        --index_path /gscratch/zlab/sewon/nplm-inference/out/ours-v1_1.3B_250B_semibalanced/train-0/$raw_file-1024-512-[0K-2000K].index \
        --tokenized_dir /gscratch/zlab/sewon/nplm-inference/out/neoX/train-0 \
        --output_dir out \
        --log_file_name out/search_config_$raw_file.out
done