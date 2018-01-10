# Pytorch language model

Language model implimented in pytorch

# Requirements

+ pytorch

# How to run

## Data sets

- train data
- vocabulary file

See [this repo](https://github.com/vintersnow/wiki-tokenize)

### Train data

format: Tokenized by `nltk.tokenized.sent_tokenize`

Example:

```
Tokenized Text .
Seconed Text .
```

### Vocabulary file
Fromat: `{word} {frequency}`

## Execute

```
mkdir ckpt runs
python train.py --num_iters 100 --store_summary --data_path 'path/to/data*' --vocab_file 'path/to/vocab'
```
