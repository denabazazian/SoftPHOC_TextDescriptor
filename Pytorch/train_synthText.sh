CUDA_VISIBLE_DEVICES=0,1,2,3 python train_synthText_fixedLR.py --backbone resnet --lr 0.00001 --weight-decay 0.00001 --workers 16 --epochs 100 --batch-size 16 --gpu-ids 0,1,2,3 --init /path/to/trained_models/deeplab/deeplab_pascalVOC/deeplab-resnet.pth.tar --checkname adam_BCE_resnet --eval-interval 1 --dataset synthText

