python train.py -ms l --data /your/imagenet-data/path/ --epochs 100 --lr-decay 0.1 --lr-mode multistep --lr-decay-epoch 30,60,90 -c checkpoints/mixnet --gpu-id 0,1,2,3,4,5,6,7
