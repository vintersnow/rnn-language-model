import queue
from threading import Thread
import time
import numpy as np
import glob
import random
import re
import torch
from torch.autograd import Variable

from collections import namedtuple

from utils import get_logger, INFO

# logger = get_logger(__name__, DEBUG)
logger = get_logger(__name__, INFO, 'logs/batcher.log')

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '_PAD'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '_UNK'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '_START'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '_STOP'  # This has a vocab id, which is used at the end of untruncated target sequences


def text_generator(data_path, single_pass):
    file_list = glob.glob(data_path)
    if len(file_list) == 0:
        logger.error('No file is found: %s', data_path)
        raise ValueError('No file is found')
    while True:
        random.shuffle(file_list)
        for file in file_list:
            with open(file, 'r') as f:
                for line in f:
                    yield 'a', line.strip()
        if single_pass:
            break


class Vocab(object):
    """単語とidをmappingするクラス"""

    def __init__(self, vocab_file, max_size):
        """
        Args:
            Vocab_file: 語彙ファイルまでのpath. 語彙ファイルの各行は"<word>
            <frequency>"という形式でfrequency順のソートされている前提.
            max_size: 使用する最大単語する（頻度順）. 0の時は全単語を使用.
        """

        self._word_to_id = {}
        self._id_to_word = {-1: ''}
        self._counter = 0

        special_tokens = [
            PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING
        ]
        for w in special_tokens:
            self._add_word(w)

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                pieces = line.split()
                if len(pieces) != 2:
                    logger.warning(
                        'Incorrect formatted line in vocabulary file: {}\n'.
                        format(line))
                    continue
                w = pieces[0]
                if w in special_tokens:
                    raise Exception(
                        'A word "{}" conflicts with special_tokens'.format(w))
                if w in self._word_to_id:
                    raise Exception('Dupilcated word: {}'.format(w))

                self._add_word(w)

                if max_size != 0 and self._counter >= max_size:
                    logger.info(
                        'Reached to max size of vocabulary: {}. Stop reading'.
                        format(max_size))
                    break
        logger.info('Finished loading vocabulary')

    def _add_word(self, word):
        self._word_to_id[word] = self._counter
        self._id_to_word[self._counter] = word
        self._counter += 1

    def word2id(self, word):
        """word(string)に対応するid(integer)を返す"""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, id):
        """id(integer)に対応するword(string)を返す"""
        if id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % id)
        return self._id_to_word[id]

    def size(self):
        return self._counter

    def pad_id(self):
        return self.word2id(PAD_TOKEN)


class Example(object):
    """Example(入力と出力のペア)のためのクラス"""

    def __init__(self, article, abstract, vocab, hps):
        """
        Args:
            article (string): 記事本文
            abstract_sentences (string): 要約文
            Vocab: Instance of Vocabulary class
            hps: hyperparameters
                max_enc_steps
                max_dec_steps
        """

        self._vocab = vocab
        self._hps = hps

        # Process the article
        article_words = article.split()  # list of strings
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        self.enc_len = len(article_words)  # length before padding
        self.enc_input = [vocab.word2id(w) for w in article_words]

        # Process the abstract
        abstract_words = abstract.split()  # list of strings
        abs_ids = [vocab.word2id(w) for w in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(
            abs_ids, hps.max_dec_steps)
        self.dec_len = len(self.dec_input)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract

    def get_dec_inp_targ_seqs(self, sequence, max_len):
        """出力からDecoderの入力文を作成する.
        Decoderの入力としてSTART_DECODINGを先頭に挿入したもの
        Decoderの出力としてSTOP_DECODINGを最後に追加したもの

        Args:
            sequence: 出力文 (数値化したもの)
            max_len: decoderの入出力の最大長
        """
        start_id = self._vocab.word2id(START_DECODING)
        stop_id = self._vocab.word2id(STOP_DECODING)

        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_input(self, max_len, pad_id):
        """decoder_inputをpad_idでmax_lenになるようにパディング"""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        """encoder_inputをpad_idでmax_lenになるようにパディング"""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)


class Batch(object):
    """Exampleを集めたミニバッチ用のクラス"""

    def __init__(self, example_list, hps, vocab):
        """
        Args:
            example_list: list of Instance of Example class
            hps: hyperparameters
            vocab: Instance of Vocabulary class
        """
        self._hps = hps
        self._vocab = vocab

        self.init_encoder_seq(example_list, hps)
        self.init_decoder_seq(example_list, hps)

    def init_encoder_seq(self, example_list, hps):
        """
        以下を初期化
        self.enc_batch:
            numpy array of shape (batch_size, <=max_enc_steps).
        self.enc_lens:
            numpy array of shape (batch_size). 各Exampleのencoderの入力長
        self.enc_padding_mask:
            numpy array of shape (batch_size, <=max_enc_steps). paddingの所だけ0
        """

        pad_id = self._vocab.word2id(PAD_TOKEN)

        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, pad_id)

        # TODO: 改善の余地あり
        # Initialize the numpy arrays
        self.enc_batch = np.zeros(
            (hps.batch_size, max_enc_seq_len), dtype=np.int64)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.ones(
            (hps.batch_size, max_enc_seq_len), dtype=np.dtype('uint8'))
        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 0
        if hps.torch:
            self.enc_batch = Variable(
                torch.from_numpy(self.enc_batch), requires_grad=False)
            self.enc_lens = Variable(
                torch.from_numpy(self.enc_lens), requires_grad=False)
            self.enc_padding_mask = Variable(
                torch.from_numpy(self.enc_padding_mask), requires_grad=False)

    def init_decoder_seq(self, example_list, hps):
        """
        以下を初期化
        self.dec_batch:
        self.target_batch:
        self.dec_lens
        self.dec_padding_mask
        """

        pad_id = self._vocab.word2id(PAD_TOKEN)

        max_dec_seq_len = max([ex.dec_len for ex in example_list])

        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_input(max_dec_seq_len, pad_id)

        # TODO: 改善の余地あり
        # Initialize the numpy arrays.
        self.dec_batch = np.zeros(
            (hps.batch_size, max_dec_seq_len), dtype=np.int64)
        self.target_batch = np.zeros(
            (hps.batch_size, max_dec_seq_len), dtype=np.int64)
        self.dec_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.dec_padding_mask = np.ones(
            (hps.batch_size, max_dec_seq_len), dtype=np.dtype('uint8'))
        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 0
        if hps.torch:
            self.dec_batch = Variable(
                torch.from_numpy(self.dec_batch), requires_grad=False)
            self.target_batch = Variable(
                torch.from_numpy(self.target_batch), requires_grad=False)
            self.dec_lens = Variable(
                torch.from_numpy(self.dec_lens), requires_grad=False)
            self.dec_padding_mask = Variable(
                torch.from_numpy(self.dec_padding_mask), requires_grad=False)

    def expand(self, use_cuda):
        '''return in tuple
        if use_cuda is true, send to GPU
        '''
        data = (self.enc_batch, self.enc_lens, self.enc_padding_mask,
                self.dec_batch, self.target_batch, self.dec_lens,
                self.dec_padding_mask)
        if use_cuda:
            data = (v.cuda() for v in data)
        return data


class Batcher(object):
    """バッチジェネレイター"""
    BATCH_QUEUE_MAX = 100

    def __init__(self,
                 data_path,
                 vocab,
                 hps,
                 single_pass,
                 generator=text_generator):
        """
        Args:
            data_path:
            vocab: Instance of Vocab class
            hps: hyperparameters
                mode: "train" "infer"
                batch_size: (integer)
                max_enc_steps
                max_dec_steps
            single_pass: Trueの場合、一度だけ全てのデータを見る
            generator: generator to generate (src, target) pair
        """

        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass
        self._generator = generator
        self._sort_by = hps.sort_by

        # minibatch用のQueue
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        # data example用のQueue
        self._example_queue = queue.Queue(
            self.BATCH_QUEUE_MAX * self._hps.batch_size)

        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 16  # num threads to fill example queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            self._bucketing_cache_size = 20  # how many batches-worth of examples to load into cache before bucketing

        # デーモンスレッドのみになった時、プロセスは終了する
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self.make_thread(self._example_q_threads, self.fill_example_queue)
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self.make_thread(self._batch_q_threads, self.fill_batch_queue)

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def make_thread(self, threads_list, target):
        threads_list.append(Thread(target=target))
        threads_list[-1].daemon = True
        threads_list[-1].start()

    def fill_example_queue(self):
        """dataファイルから読み込んで、exampleキューに入れる"""

        text_gn = self._generator(self._data_path, self._single_pass)
        clean_sent = re.compile(r'{}|{}'.format(SENTENCE_START, SENTENCE_END))

        while True:
            try:
                (article, abstract) = next(text_gn)
            except StopIteration:  # No more example
                logger.info('No more example')
                if self._single_pass:
                    logger.info('Successfully finished single pass mode')
                    self._finished_reading = True
                    break
                else:
                    raise Exception(
                        'example_gn is out of data but single pass mode is off'
                    )

            # Erase <s> </s>
            article = clean_sent.sub('', article)
            abstract = clean_sent.sub('', abstract)
            example = Example(article, abstract, self._vocab, self._hps)
            self._example_queue.put(example)

    def fill_batch_queue(self):
        """exampleキューから収得して、入力長でソートした後にBatchを作成してbatchキューに入れる"""
        batch_size = self._hps.batch_size
        while True:
            if self._hps.mode != 'infer':
                num_example = batch_size * self._bucketing_cache_size
                inputs = [
                    self._example_queue.get() for _ in range(num_example)
                ]
                # input lengthでソート
                if self._sort_by == 'enc':
                    inputs = sorted(
                        inputs, key=lambda inp: inp.enc_len, reverse=True)
                elif self._sort_by == 'dec':
                    inputs = sorted(
                        inputs, key=lambda inp: inp.dec_len, reverse=True)

                batches = [
                    inputs[i:i + batch_size]
                    for i in range(0, len(inputs), batch_size)
                ]
                for b in batches:
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))
            # TODO: beam search時は少し違う？
            else:
                logger.error('Not supported mode: %s' % self._hps.mode)
                raise ValueError('Not supported mode: %s' % self._hps.mode)

    def next_batch(self):
        """Return a Batch from batch queue (or None when finished in single pass mode)
        """
        if self._batch_queue.qsize() == 0:
            logger.warning(
                'Batch input queue is empty when calling next_batch'
                '. Batch queue size={}, Example queue size={}'.format(
                    self._batch_queue.qsize(), self._example_queue.qsize()))

            if self._single_pass and self._finished_reading:
                logger.info("Finished reading dataset in single_pass mode.")
                return None

        return self._batch_queue.get()

    def __iter__(self):
        while True:
            batch = self.next_batch()
            if batch is not None:
                yield batch
            else:
                break

    def watch_threads(self):
        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    logger.error(
                        'Found example queue thread dead. Restarting.')
                    self.make_thread(self._example_q_threads,
                                     self.fill_example_queue)
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    logger.error('Found batch queue thread dead. Restarting.')
                    self.make_thread(self._batch_q_threads,
                                     self.fill_batch_queue)


def restore_text(data, vocab):
    """
    Args:
        data: (sequence_length) list or ndarray include word id (integer)
        vocab:
    """
    text = ' '.join([vocab.id2word(id) for id in data])
    return text


if __name__ == '__main__':
    # make_vocab_file(pair_generator, 'data/eng-fra.txt')
    # raise ValueError('')

    print('----')
    # vocab = Vocab('data/vocab', 50000)
    vocab = Vocab('data/vocab', 10000)
    print('vocab size', vocab.size())

    hps_dict = {
        'batch_size': 32,
        'mode': 'train',
        'max_enc_steps': 100,
        'max_dec_steps': 100,
        'sort_by': 'dec',
        'torch': False,
        'single_pass': True
    }
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    # batcher = Batcher('data/test.bin', vocab, hps, False)
    batcher = Batcher('data/output*', vocab, hps, hps.single_pass, text_generator)

    for step in range(1000):
        time.sleep(0.5)
        batch = batcher.next_batch()
        print(batch.enc_batch.shape, batch.dec_batch.shape,
              batch.target_batch.shape)
        for id in batch.enc_batch[0]:
            print(vocab.id2word(id), end=' ')
        for id in batch.target_batch[0]:
            print(vocab.id2word(id), end=' ')
        break
