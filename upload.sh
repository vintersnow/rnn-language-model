#!/bin/sh

rsync -a --progress \
  --exclude __pycache__\
  --exclude utils \
  --exclude .git \
  --exclude data \
  --exclude *.log \
  --exclude ckpt \
  --exclude logs \
  --exclude runs \
  --exclude *.json \
  --exclude .ropeproject \
  -e ssh ~/Projects/ut/nlp/lang-model kobe:~/projects/
