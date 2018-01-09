from argparse import ArgumentParser
import torch


parser = ArgumentParser('python train.py')

# Data
parser.add_argument('--vocab_size', type=int, default=10000,
                    help='max number of vocabulary')
parser.add_argument('--vocab_file', type=str, default='data/vocab',
                    help='path to vocabulary file')
parser.add_argument('--data_path', type=str, default='data/output-*',
                    help='path to data file or directory')
parser.add_argument('--single_pass', action='store_true',
                    help='If the flag is setted, applay example only once')

# Model
parser.add_argument('--max_enc_steps', type=int, default=1000,
                    help='max length for encoder')
parser.add_argument('--max_dec_steps', type=int, default=100,
                    help='max length for decoder')
parser.add_argument('--mode', type=str, default='train',
                    help='')
parser.add_argument('--embed_size', type=int, default=256,
                    help='size of word embedding')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='size of hidden units')
# parser.add_argument('--output_size', type=int, default=256,
#                     help='size of decoder output units')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of leyers for the rnn')
# parser.add_argument('--no_attention', action='store_true',
#                     help='no attention in decoding')
# parser.add_argument('--hidden_trans', action='store_true',
#                     help='translate hidden units for decoder from encoder')

# Training
parser.add_argument('--num_iters', type=int, default=100,
                    help='number of training iterations')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--init_lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=2.0,
                    help='gradient clipping norm size')
parser.add_argument('--lr_decay_step', type=int, default=5000,
                    help='decay the learning rate each `steps`')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout ratio for encoder and decoder')
parser.add_argument('--opt', type=str, default='adam',
                    help='optimizer method')
parser.add_argument('--start_step', type=int, default=1,
                    help='initial step number')

# Summary and checkpoint
parser.add_argument('--summary_steps', type=int, default=100,
                    help='interval for reporting summary')
parser.add_argument('--ckpt_steps', type=int, default=2000,
                    help='interval for reporting summary')
parser.add_argument('--store_summary', action='store_true',
                    help='store summary')
parser.add_argument('--ckpt_path', type=str, default='ckpt',
                    help='directory path to save model parameters')
parser.add_argument('--ckpt_name', type=str, default='seq2seq',
                    help='checkpoint file name (`{ckpt_name}-{step}.ckpt`)')
parser.add_argument('--restore', type=str, default=None,
                    help='file path for restore parameters')

# Other
parser.add_argument('--use_cuda', action='store_true',
                    help='use cuda')

hps = parser.parse_args()

hps.use_cuda = hps.use_cuda and torch.cuda.is_available()
