CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone resnet --lr 0.004 --workers 8 --epochs 50 --batch-size 3 --gpu-ids 0,1,2,3 --checkname deeplab-resnet --eval-interval 1 --dataset icdar
