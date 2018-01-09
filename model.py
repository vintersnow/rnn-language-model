import torch
from torch import nn
from hyperparams import hps
from torch.nn.utils.rnn import (pad_packed_sequence, pack_padded_sequence)

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
            dropout=hps.dropout
        )

        self._out_layer = nn.Linear(hps.output_size, vocab_size)
        self._s = nn.LogSoftmax(2)

    def forward(self,
                inputs,
                inputs_lens):
        '''
        B: batch_size
        U: unite_size (e.g. hidden_size)
        T: seqence lengths
        L: num of layers

        Args:
            inputs (Variable): size=(B*T1)
            inputs_lens (list[int]): size=(B)
            # inp_pad (ByteTensor): size=(B*T). set 1 for encoder padding
            # dec_inputs (Variable): size(B*T2)
            # dec_input_lens (LongTensor): size=(B)
        returns:
            output (Variable): (B*T2*U2). U2=U ~ 3U
                attentionを使うか、エンコーダをbidirectionalしているかによって変わる
        '''

        embd = self._embd_layer(inputs)
        if hps.use_cuda:
            seq_lens = inputs_lens.cpu().data.numpy()
        else:
            seq_lens = inputs_lens.data.numpy()

        # padding部分を無視するためにpackする
        packed = pack_padded_sequence(embd, seq_lens, batch_first=True)
        output, hidden = self._rnn(packed)
        # [0.0]*max(hidden_size)でpaddingし直す
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = self._out_layer(output)  # (B*T*Vocab)
        output = self._s(output)  # logsoftmax

        return output
