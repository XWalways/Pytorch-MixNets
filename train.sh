CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 train_apex.py -ms s --data /data0/imagenet --train-batch 384 --test-batch 512 --epochs 100 --lr 0.2 --lr-mode cosine -c checkpoints/mixnet --logdir ./logs/mixnet
#--lr-decay 0.1 --lr-mode multistep --lr-decay-epoch 30,60,90
