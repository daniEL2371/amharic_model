#!/bin/bash

export PATH="/home/itsc/anaconda3/bin:$PATH"
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source /home/itsc/.bashrc
cd /home/itsc/am_model
/home/itsc/.conda/envs/gputf/bin/python ./train_class_pred.py
/home/itsc/.conda/envs/gputf/bin/python ./train_char_pred.py



