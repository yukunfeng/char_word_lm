set -x

gpu_id="0"

# data_names=(vi zh ja pt en ms he)
data_names=(ar de cs es et ru fi)
for data_name in "${data_names[@]}"
do
  data_path="/home/lr/yukun/common_corpus/data/50lm/$data_name"

  # Char-BiLSTM-LSTM in the paper
  log="char.$data_name.log"
  save_model="char.en.model"
  python -u main.py --device "cuda:$gpu_id" --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-LSTM" \
  --combination '0 1 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log

  # Word-LSTM in the paper
  log="word.$data_name.log"
  save_model="word.en.model"
  python -u main.py --device "cuda:$gpu_id" --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Word-LSTM"  \
  --combination '1 0 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log

  # Char-BiLSTM-add-Word-LSTM(g= 0.5,n= 1) in the paper
  log="char-word.$data_name.log"
  save_model="char-word.en.model"
  python -u main.py --device "cuda:$gpu_id" --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-add-Word-LSTM-Word(g= 0.5,n= 2)"  \
  --combination '1 1 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log

  # Char-BiLSTM-add-Word-LSTM-Word(g= 0.5,n= 1) in the paper
  log="char-word-word.$data_name.log"
  save_model="char-word-word.en.model"
  python -u main.py --device "cuda:$gpu_id" --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-add-Word-LSTM-Word(g= 0.5,n= 2)"  \
  --combination '1 1 0.5' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650  >> $log

done

