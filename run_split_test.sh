#!/usr/bin/env bash

set -x

# data_names=(vi zh ja pt en ms he ar de cs es et ru fi)
data_names=(kim-cs)
for data_name in "${data_names[@]}"
do
    python split_test.py \
        --input_path /home/lr/yukun/common_corpus/data/50lm/$data_name/train.txt
    
done
