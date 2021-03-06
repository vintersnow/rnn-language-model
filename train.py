from model import Model
import torch
from torch import nn
from hyperparams import hps
from tensorboardX import SummaryWriter
from data_batcher import Batcher, Vocab
from utils import (get_logger, DEBUG, OneLinePrint, optimzier, Saver, Timer)
import numpy as np

logger = get_logger(__name__, DEBUG)


def train():
    olp = OneLinePrint()

    logger.info('start building batch data')

    vocab = Vocab(hps.vocab_file, hps.vocab_size)
    batcher = Batcher(hps.data_path, vocab, hps, hps.single_pass)

    logger.info('end building batch data')
    logger.info('vocab size: %s' % vocab.size())

    criterion = nn.NLLLoss(ignore_index=vocab.pad_id())

    model = Model(vocab, hps)
    if hps.use_cuda:
        model = model.cuda()
    if hps.restore:
        model.load_state_dict(torch.load(hps.restore))

    opt = optimzier(hps.opt, model.parameters())

    if hps.ckpt_name != '':
        saver = Saver(hps.ckpt_path, hps.ckpt_name, model)

    # for store summary
    if hps.store_summary:
        writer = SummaryWriter(comment='_' + hps.ckpt_name)

    # loss_sum = 0

    logger.info('----Start training----')
    timer = Timer()
    timer.start()
    for step in range(hps.start_step, hps.num_iters + 1):
        # # Decay learning rate
        # if step % hps.lr_decay_step == 0:
        #     olp.write(
        #         'decay learning rate to %f' % decay_lr(opt, step))

        # Forward -------------------------------------------------------------
        opt.zero_grad()

        batch = batcher.next_batch()
        (inputs, inp_lens, inp_pad, dec_inps, targets, dec_lens,
         dec_pad) = batch.expand(hps.use_cuda)

        outputs = model(dec_inps, dec_lens)  # output: (B*T*(1~3)U)
        loss = criterion(outputs.view(-1, vocab.size()), targets.view(-1))

        # Backward ------------------------------------------------------------
        loss.backward()
        # gradient clipping
        global_norm = nn.utils.clip_grad_norm(model.parameters(), hps.clip)
        opt.step()

        # loss_sum += loss.data[0]

        # Utils ---------------------------------------------------------------
        # save checkpoint
        if step % hps.ckpt_steps == 0 and hps.ckpt_name != '':
            saver.save(step, loss.data[0])
            olp.write('save checkpoint (step=%d)\n' % step)

        # print the train loss and ppl
        ppl = np.exp(loss.data[0])
        olp.write('step %s train loss: %f, ppl: %8.2f' %
                  (step, loss.data[0], ppl))
        olp.flush()

        # store summary
        if hps.store_summary and (step - 1) % hps.summary_steps == 0:
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('ppl', ppl, step)
            writer.add_scalar('global_norm', global_norm, step)
            if step - 1 != 0:
                lap_time, _ = timer.lap('summary')
                steps = hps.summary_steps
                writer.add_scalar('avg time/step', lap_time / steps, step)

        # print output and target
        # if step % hps.summary_steps == 0:
        #     logger.info('\nstep:%d~%d avg loss: %f', step - hps.summary_steps,
        #                 step, loss_sum / hps.summary_steps)
        #     loss_sum = 0

    if hps.store_summary:
        writer.close()


if __name__ == '__main__':
    train()
