set -x

gpu_id="0"
log="log2"
data_path="./data/en/"
save_model="$(basename $data_path).save"

# Char-BiLSTM-add-Word-LSTM-Word(g= 0.5,n=1) in the paper
# python -u main.py --device "cuda:$gpu_id" --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm  \
# --combination '1 1 0.5' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log

# Char-BiLSTM-LSTM in the paper
# python -u main.py --device "cuda:$gpu_id" --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-LSTM" \
# --combination '0 1 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log

# Word-LSTM in the paper
# python -u main.py --device "cuda:$gpu_id" --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Word-LSTM"  \
# --combination '1 0 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log

# Char-BiLSTM-gate-Word-LSTM in the paper
python -u main.py --device "cuda:$gpu_id" --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-gate-Word-LSTM"  \
--combination '-1 -1 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log

# Char-BiLSTM-cat-Word-LSTM in the paper
python -u main.py --device "cuda:$gpu_id" --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-cat-Word-LSTM" \
--combination '-2 -2 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log

# Char-BiLSTM-add-Word-LSTM-Word(g= 0.5,n= 2) in the paper
# python -u main.py --device "cuda:$gpu_id" --word_out_num 2 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-add-Word-LSTM-Word(g= 0.5,n= 2)"  \
# --combination '1 1 0.5' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log
