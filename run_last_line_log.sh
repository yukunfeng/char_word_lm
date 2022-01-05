#!/usr/bin/env bash


data_names=(vi zh ja pt en ms he ar de cs es et ru fi)
for data_name in "${data_names[@]}"
do
    tail -n 1 char.$data_name.log | python last_line_log.py
    tail -n 1 word.$data_name.log | python last_line_log.py
    tail -n 1 char-word.$data_name.log | python last_line_log.py
    tail -n 1 char-word-word.$data_name.log | python last_line_log.py
done
