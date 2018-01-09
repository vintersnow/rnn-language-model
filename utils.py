from logging import (getLogger, StreamHandler, FileHandler, CRITICAL, ERROR,
                     WARNING, INFO, DEBUG)
import sys
import time
from hyperparams import hps
import torch
from torch.autograd import Variable
from os import path, mkdir, rename
import itertools
# from pythonrouge.pythonrouge import Pythonrouge


def get_logger(name, level=INFO, output_file=None):
    """
    Args:
        naame: getLoggerに渡す変数.
        level: ログ出力レベル. デフォルトはINFO.
        output_file: ログ出力ファイル. Noneの場合は標準出力. デフォルトはNone.
    """
    levels = [CRITICAL, ERROR, WARNING, INFO, DEBUG]
    if level not in levels:
        raise ValueError('"level" should be {}. Received "{}" insted.'.format(
            levels, level))

    logger = getLogger(name)

    handler = StreamHandler() if output_file is None else FileHandler(
        output_file, 'a+', encoding='utf-8')
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


class OneLinePrint(object):
    def __init__(self, new_line_marker=' $ '):
        self._mark = new_line_marker
        self._pstr = ''

    def write(self, str):
        if self._pstr == '':
            self._pstr += '\r\033[K'
        else:
            self._pstr += self._mark
        self._pstr += str.replace('\n', self._mark)

    def flush(self):
        sys.stdout.write(self._pstr)
        sys.stdout.flush()
        self._pstr = ''


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def start(self):
        self.begin = self.prev = time.time()

    def lap(self, comment=''):
        now = time.time()
        lap = (now - self.prev) * 1e3
        all = (now - self.begin) * 1e3
        self.prev = now
        if self.verbose:
            print('elapsed time: %f ms. lap: %f ms. %s' % (all, lap, comment))
        return lap, all


# def rouge_score(summary, reference):
#         rouge = Pythonrouge(
#             summary_file_exist=False,
#             summary=summary,
#             reference=reference,
#             n_gram=3,
#             ROUGE_SU4=True,
#             ROUGE_L=False,
#             recall_only=False,
#             stemming=True,
#             stopwords=True,
#             word_level=True,
#             length_limit=True,
#             length=50,
#             use_cf=False,
#             cf=95,
#             scoring_formula='average',
#             resampling=True,
#             samples=1000,
#             favor=True,
#             p=0.5)
#         return rouge.calc_score()

###############################################################################
# Torch utils


def zero_var(*size):
    var = Variable(torch.zeros(*size))
    if hps.use_cuda:
        var = var.cuda()
    return var


def optimzier(method, params, **keys):
    if method.lower() == 'adam':
        return torch.optim.Adam(params, lr=hps.init_lr, **keys)
    elif method.lower() == 'adagrad':
        return torch.optim.Adagrad(params, lr=hps.init_lr, **keys)
    else:
        raise ValueError('Unkonw optimzier method: %s', method)


def tonp(tens):
    if hps.use_cuda:
        return tens.cpu().data.numpy()
    else:
        return tens.data.numpy()


def decay_lr(opt, step):
    assert isinstance(opt, torch.optim.Optimizer)
    lr = hps.decay_rate**(step // hps.lr_decay_step) * hps.init_lr
    for p in opt.param_groups:
        p['lr'] = lr
    return lr


def save(model, step):
    file_name = '%s-%d.ckpt' % (hps.ckpt_name, step)
    file_path = path.join(hps.ckpt_path, file_name)
    torch.save(model.state_dict(), file_path)


class Saver(object):
    def __init__(self, ckpt_path, name, model):
        self._ckpt_path = path.join(ckpt_path, name)
        self._name = name
        self._model = model
        if path.isdir(self._ckpt_path):
            for i in itertools.count():
                bk_path = path.join(ckpt_path, name + '.' + str(i))
                if not path.isdir(bk_path):
                    rename(self._ckpt_path, bk_path)
                    break
        mkdir(self._ckpt_path)

        with open(path.join(self._ckpt_path, self._name + '.model'), 'w') as f:
            f.write(str(model))

    def save(self, step, loss):
        file_name = '%s_step-%d_loss-%.3f.ckpt' % (self._name, step, loss)
        file_path = path.join(self._ckpt_path, file_name)
        torch.save(self._model.state_dict(), file_path)


# if __name__ == '__main__':
#     model = torch.nn.Linear(10, 10)
#     saver = Saver('ckpt', 'test', model)
#     saver.save(10)
#     saver.save(11)
#     saver.save(13)
