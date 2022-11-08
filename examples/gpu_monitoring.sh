#!/bin/bash
sudo add-apt-repository ppa:flexiondotorg/nvtop -y > /dev/null
sudo apt install nvtop > /dev/null

nohup python train.py experiments=IM_190722_vwer3f23_oneshotlearning.yaml optimizers=auto accelerator=gpu +devices=1 &
PID=$!
sudo nvtop
kill $PID