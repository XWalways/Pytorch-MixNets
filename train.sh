CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 train.py -ms s --data /data0/imagenet --train-batch 512 --test-batch 512 --epochs 100 --lr 0.2 --lr-mode cosine -c checkpoints/mixnet --logdir ./logs/mixnet
#--lr-decay 0.1 --lr-mode multistep --lr-decay-epoch 30,60,90
