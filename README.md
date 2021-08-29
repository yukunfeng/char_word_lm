## Injecting Word-level Information into Character-aware NLMs
Pytorch Implementation of CoNLL 2019 paper [A Simple and Effective Method for Injecting Word-level Information into Character-aware Neural Language Models](https://www.aclweb.org/anthology/K19-1086.pdf)
 
## Requirements
- Python version >= 3.5
- Pytorch version 0.4.0

## Datasets
Originally downloaded from [here](http://people.ds.cam.ac.uk/dsg40/lmmrl.html) from [this
paper](https://www.aclweb.org/anthology/Q18-1032.pdf). Currently the link seems broken and I have
uploaded one English dataset for testing under 'data' directory. 

## Usage

Some main options are explained as follows. See more using `python main.py -h`
```

 -h, --help            show this help message and exit
  --data DATA           location of the data directory. train.txt, valid.txt
                        and test.txt are put here
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --word_out_num WORD_OUT_NUM
                        number of word injected into the softmax function. If
                        it's 2, current word and prevous word are injected.
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --save SAVE           path to save the final model
  --combination COMBINATION
                        the format is "a b c"
                        a*word + b*bilstm as input to lstm
                        lstm_output + c*word as input to softmax function
                        if a=-1 and b=-1, the model will use gating mechanism to do the combination
                        if a=-2 and b=-2, the model will use concat method to do the combination
                        if c=-1, the model will use a gating mechanism on word when injected into softmax
  --max_gram_n MAX_GRAM_N
                        character n-gram. We use 3 at default.
  --use_bilstm          whether use bilstm. If not, there is only word-level
                        information
  --input_freq INPUT_FREQ
                        freq threshould for input word
  --note NOTE           extra note in final one-line result output
```

## Run experiments reported in the paper

Run model Char-BiLSTM-add-Word-LSTM-Word(g=0.5,n=1) in the paper:
```
python -u main.py --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm  \
 --combination '1 1 0.5' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650
```

Run model Char-BiLSTM-LSTM in the paper:
```
python -u main.py --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-LSTM" \
 --combination '0 1 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650
```

Run Word-LSTM in the paper:
```
python -u main.py --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Word-LSTM"  \
 --combination '1 0 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650
```

Run Char-BiLSTM-gate-Word-LSTM in the paper:
```
python -u main.py --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-gate-Word-LSTM"  \
--combination '-1 -1 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650
```

Run Char-BiLSTM-cat-Word-LSTM in the paper:
```
python -u main.py --word_out_num 1 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-cat-Word-LSTM" \
--combination '-2 -2 0' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650
```

Run Char-BiLSTM-add-Word-LSTM-Word(g=0.5,n=2) in the paper:
```
python -u main.py --word_out_num 2 --input_freq 1 --max_gram_n 3 --use_bilstm --note "Char-BiLSTM-add-Word-LSTM-Word(g= 0.5,n= 2)"  \
 --combination '1 1 0.5' --save $save_model --dropout 0.5  --data $data_path --epoch 40 --emsize 650 --nhid 650
```
