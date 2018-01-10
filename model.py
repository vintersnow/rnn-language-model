import torch
from torch import nn
from hyperparams import hps
from torch.nn.utils.rnn import (pad_packed_sequence, pack_padded_sequence)
from torch.autograd import Variable
from utils import zero_var
from data_batcher import START_DECODING, STOP_DECODING

dtype = torch.FloatTensor


class Model(nn.Module):
    def __init__(self, vocab, hps, bidirectional=True, hidden_trans=False):
        super(Model, self).__init__()

        self._hps = hps
        self._vocab = vocab
        self._vocab_size = vocab_size = vocab.size()

        self._embd_layer = nn.Embedding(
            vocab_size, hps.embed_size, padding_idx=vocab.pad_id())

        self._rnn = nn.GRU(
            self._embd_layer.embedding_dim,
            hps.hidden_size,
            hps.num_layers,
            bidirectional=False,
            dropout=hps.dropout)

        self._out_layer = nn.Linear(hps.hidden_size, vocab_size)
        self._s = nn.LogSoftmax(2)

    def forward(self, inputs, inputs_lens, mode='train'):
        '''
        B: batch_size
        U: unite_size (e.g. hidden_size)
        T: seqence lengths
        L: num of layers

        Args:
            inputs (Variable): size=(B*T)
            inputs_lens (list[int]): size=(B)
        returns:
            output (Variable): (B*T*V). mode='train'の場合 
            output (Variable): (T*V). mode='infer'の場合.
        '''

        if mode == 'train':
            embd = self._embd_layer(inputs)
            if hps.use_cuda:
                seq_lens = inputs_lens.cpu().data.numpy()
            else:
                seq_lens = inputs_lens.data.numpy()

            # padding部分を無視するためにpackする
            packed = pack_padded_sequence(embd, seq_lens, batch_first=True)
            output, hidden = self._rnn(packed)
            # [0.0]*max(hidden_size)でpaddingし直す
            outputs, _ = pad_packed_sequence(output, batch_first=True)

            outputs = self._out_layer(outputs)  # (B*T*Vocab)
            outputs = self._s(outputs)  # logsoftmax
            return outputs
        elif mode == 'infer':
            max_steps = self._hps.max_steps
            outputs = zero_var(max_steps, self._vocab_size)
            start_id = self._vocab.word2id(START_DECODING)
            stop_id = self._vocab.word2id(STOP_DECODING)
            input = Variable(torch.LongTensor([start_id]))
            if hps.use_cuda:
                input = input.cuda()
            hidden = zero_var(self._hps.num_layers, 1, self._hps.hidden_size)
            # hidden.data.uniform_(-1e-1, 1e-1)
            for i in range(max_steps):
                embd = self._embd_layer(input.unsqueeze(0))  # (1*U)
                output, hidden = self._rnn(embd, hidden)
                output = self._out_layer(output)  # (1*1*Vocab)
                output = self._s(output).squeeze()  # (1*1*V) -> (V)
                outputs[i] = output
                _, input = torch.max(output, 0)
                if input.data[0] == stop_id:
                    break
            return outputs
        else:
            raise ValueError('Unknown mode: %s' % mode)

