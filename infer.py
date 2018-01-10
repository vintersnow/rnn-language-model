from model import Model
import torch
from hyperparams import hps
from data_batcher import Vocab, restore_text
from utils import (get_logger, DEBUG, tonp, Timer)

logger = get_logger(__name__, DEBUG)


def train():
    logger.info('start building vocab data')

    vocab = Vocab(hps.vocab_file, hps.vocab_size)

    logger.info('end building vocab data')
    logger.info('vocab size: %s' % vocab.size())

    model = Model(vocab, hps)
    if hps.use_cuda:
        model = model.cuda()
    if hps.restore is not None:
        # raise ValueError('Noe data to restore')
        model.load_state_dict(torch.load(hps.restore))

    logger.info('----Start training----')
    timer = Timer()
    timer.start()
    for step in range(hps.start_step, hps.num_iters + 1):

        # Forward -------------------------------------------------------------
        outputs = model(None, None, 'infer')
        _, pred = torch.max(outputs, 1)
        pred = tonp(pred)
        logger.info('pred: %s' % restore_text(pred, vocab, [vocab.pad_id()]))

    lap, _ = timer.lap('end')
    print('pred time: %f', lap)


if __name__ == '__main__':
    train()
