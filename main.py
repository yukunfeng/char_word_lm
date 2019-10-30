# coding: utf-8
import argparse
import time
import datetime
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torchtext

import data
import model

class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

parser = argparse.ArgumentParser(
    description='PyTorch RNN/LSTM Language Model',
    formatter_class=SmartFormatter
)
parser.add_argument('--data', type=str, default='./data/en/',
                    help='location of the data directory. train.txt, valid.txt and test.txt are put here')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--word_out_num', type=int, default=1,
                    help='''number of word injected into the softmax function. If it's 2, current
                    word and prevous word are injected.''')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
# not used in word-level model to make a fair setting
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='cuda')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--combination', type=str, default="a b c",
                    help="""R|the format is "a b c"
a*word + b*bilstm as input to lstm
lstm_output + c*word as input to softmax function
if a=-1 and b=0, the model will use gating mechanism to do the combination
if a=-2 and b=0, the model will use concat method to do the combination
if c=-1, the model will use a gating mechanism on word when injected into softmax""")

parser.add_argument('--max_gram_n', type=int, default=3,
                    help='character n-gram. We use 3 at default.')
parser.add_argument('--use_bilstm', action='store_true',
                    help='whether use bilstm. If not, there is only word-level information')
parser.add_argument('--input_freq', type=int, default=None,
                    help='freq threshould for input word')
parser.add_argument('--note', type=str, default="",
                    help='extra note in final one-line result output')
args = parser.parse_args()
args.combination = args.combination.split()
args.weights = [float(i) for i in args.combination]

print(args)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    pass
    #  if not args.cuda:
        #  print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#  device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device(args.device)



###############################################################################
# Load data
###############################################################################

input_extra_unk = "<input_extra_unk>"
if args.tied:
    input_extra_unk = None
corpus = data.Corpus(
    args.data,
    use_ngram=True,
    max_gram_n=args.max_gram_n,
    input_freq=args.input_freq,
    input_extra_unk=input_extra_unk
)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

##########################
#  load pre-trained emb  #
##########################


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def batchify_ngram(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, nbatch, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# get fixed input data
fixed_train_data = batchify(corpus.train_fixed, args.batch_size)
fixed_val_data = batchify(corpus.valid_fixed, eval_batch_size)
fixed_test_data = batchify(corpus.test_fixed, eval_batch_size)


ngram_train = batchify_ngram(corpus.ngram_train, args.batch_size)
ngram_val = batchify_ngram(corpus.ngram_valid, eval_batch_size)
ngram_test = batchify_ngram(corpus.ngram_test, eval_batch_size)

ngram_train_len = None
ngram_valid_len = None
ngram_test_len = None
#  if args.use_bilstm:
ngram_train_len = batchify(corpus.ngram_train_len, args.batch_size)
ngram_valid_len = batchify(corpus.ngram_valid_len, eval_batch_size)
ngram_test_len = batchify(corpus.ngram_test_len, eval_batch_size)

###############################################################################
# Build the model
###############################################################################


ntokens = len(corpus.dictionary)
input_tokens = len(corpus.input_dict.idx2word)
input_unseen_tag_index = corpus.input_unseen_idx
ngram_num = len(corpus.char_ngrams.chars2idx)

model = model.RNNModel(
    args.model,
    input_tokens,
    input_unseen_tag_index,
    ntokens,
    args.emsize,
    args.nhid,
    args.nlayers,
    args.dropout,
    args.tied,
    ngram_num,
    corpus.char_ngrams.pad_index,
    args.use_bilstm,
    args.weights
).to(device)

model.word_out_num = args.word_out_num
# print model's information
note_for_model_info = f"{args.data} {args.note}"
model.model_info(note_for_model_info)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_batch_ngram(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    return data, None


def evaluate(
    data_source, fixed_data_source, 
    data_source_ngram, data_source_ngram_len):

    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            _, targets = get_batch(data_source, i)
            data, _ = get_batch(fixed_data_source, i)
            ngram_data, _ = get_batch_ngram(data_source_ngram, i)
            ngram_data_len = None
            #  if args.use_bilstm:
            ngram_data_len, _ = get_batch(data_source_ngram_len, i)

            output, hidden = model(data, ngram_data, ngram_data_len, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)

    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        _, targets = get_batch(train_data, i)
        data, _ = get_batch(fixed_train_data, i)
        ngram_data, _ = get_batch_ngram(ngram_train, i)
        ngram_data_len = None
        #  if args.use_bilstm:
        ngram_data_len, _ = get_batch(ngram_train_len, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data,  ngram_data, ngram_data_len, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            #  print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    #  'loss {:5.2f} | ppl {:8.2f}'.format(
                #  epoch, batch, len(train_data) // args.bptt, lr,
                #  elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
all_train_start_time = datetime.datetime.now()
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data, fixed_val_data, ngram_val, ngram_valid_len)

        #  print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} lr:{:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss), lr))
        #  print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            #  with open(args.save, 'wb') as f:
                #  torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
        #  if epoch % 4 == 0:
            #  test_loss = evaluate(test_data, ngram_test, False)
            #  test_loss = evaluate(test_data, fixed_test_data, ngram_test, ngram_test_len)
            #  print(f"test ppl : {math.exp(test_loss)}")
        if lr < 0.05:
            break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

all_train_end_time = datetime.datetime.now()
all_train_time = f"{all_train_end_time - all_train_start_time}"
print(f"all training time elapsed: {all_train_time}")
print(f"training time per epoch: {(all_train_end_time - all_train_start_time) / args.epochs}")

# Load the best saved model.
#  with open(args.save, 'rb') as f:
    #  model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    #  model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data, fixed_test_data, ngram_test, ngram_test_len)
print('=' * 89)
print('| End of training | test loss {:5.2f} | best_val_ppl {:8.2f} test ppl {:8.2f}'.format(
    test_loss, math.exp(best_val_loss), math.exp(test_loss)))
print('=' * 89)
print(f"final_out: {args.data} {args.note} {math.exp(best_val_loss)} {math.exp(test_loss)}")
# run logging files

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
