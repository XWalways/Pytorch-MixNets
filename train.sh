python train.py -ms s --data /home/data/.mxnet/datasets/imagenet --epochs 100 --lr-decay 0.1 --lr-mode multistep --lr-decay-epoch 30,60,90 -c checkpoints/mixnet --logdir ./logs/mixnet --gpu-id 0,1
