Deep lab v3+ Mobilenet
CUDA_VISIBLE_DEVICES=3 python3 train.py --base_model deeplabv3+_mobilenet --test 1 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 24  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > test.txt

CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_mobilenet --test 1 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > sample_fz_mobilenet_nd_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_mobilenet  --test 2 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > sample_fz_mobilenet_dz_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_mobilenet  --test 3 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > sample_fz_mobilenet_fdd_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_mobilenet  --test 4 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > sample_fz_mobilenet_fd_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_mobilenet  --test 5 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > sample_fz_mobilenet_fz_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_mobilenet  --test 6 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > sample_fz_mobilenet_full_anonymous6295_result.txt


--resume /home/anonymous6295/scratch/models/deeplabv3_r101-d8_512x1024_40k_cityscapes_20200605_012241-7fd3f799.pth 
CUDA_LAUNCH_BLOCKING=1

 CUDA_VISIBLE_DEVICES=3 python3 train.py   --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar  --backbone resnet --lr 0.001 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-9  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > resnet_101_day_anonymous6295_result.txt
--resume /home/anonymous6295/scratch/models/deeplabv3_r101-d8_512x1024_40k_cityscapes_20200605_012241-7fd3f799.pth 

Deep lab v3+ Resnet 101 
CUDA_VISIBLE_DEVICES=0 python3 train.py   --backbone resnet --lr 0.001 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-9  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > refinenet_101_nd_anonymous6295_result.txt
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python3 train.py --base_model deeplabv3+_resnet101 --test 0 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 > sample_fz_resnet_101_day_anonymous6295_result.txt

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_resnet101 --test 2 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar --backbone resnet --lr 0.01 --workers 6 --batch_size 6  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >test_sample.txt
CUDA_VISIBLE_DEVICES=3 python3 train_sample.py  --base_model deeplabv3+_resnet101 --test 2 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar  --backbone resnet --lr 0.001 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >test.txt
CUDA_VISIBLE_DEVICES=3 python3 train_sample.py  --base_model deeplabv3+_resnet101 --test 3 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >sample_fz_resnet_101_fdd_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_resnet101 --test 4 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar  --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >sample_fz_resnet_101_fd_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_resnet101 --test 5 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar  --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >sample_fz_resnet_101_fz_anonymous6295_result.txt
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python3 train_sample.py --base_model deeplabv3+_resnet101 --test 6 --resume /home/anonymous6295/scratch/models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >sample_fz_resnet_101_full_anonymous6295_result.txt


Deep labv3+ Resnet 50
CUDA_VISIBLE_DEVICES=0 python3 train.py   --backbone resnet --lr 0.001 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-9  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100    > refinenet_101_nd_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train.py --base_model deeplabv3plus_resnet50  --test 1 --resume /home/anonymous6295/scratch/models/deeplabv3plus_r50-d8_769x769_80k_cityscapes_20200606_210233-0e9dfdc4.pth  --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >fz_resnet_50_day_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train.py --base_model deeplabv3_resnet50 --test 1 --resume /home/anonymous6295/scratch/models/deeplabv3/deeplabv3plus_r50-d8_769x769_80k_cityscapes_20200606_210233-0e9dfdc4.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >fz_resnet_50_nd_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train.py  --base_model deeplabv3plus_resnet50 --test 2 --resume /home/anonymous6295/scratch/models/deeplabv3/best_deeplabv3plus_resnet50_voc_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >fz_resnet_50_dz_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train.py  --base_model deeplabv3plus_resnet50 --test 3 --resume /home/anonymous6295/scratch/models/deeplabv3/best_deeplabv3plus_resnet50_voc_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >fz_resnet_50_fdd_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train.py --base_model deeplabv3plus_resnet50 --test 4 --resume /home/anonymous6295/scratch/models/deeplabv3/best_deeplabv3plus_resnet50_voc_os16.pth  --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >fz_resnet_50_fd_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train.py --base_model deeplabv3plus_resnet50 --test 5 --resume /home/anonymous6295/scratch/models/deeplabv3/best_deeplabv3plus_resnet50_voc_os16.pth --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >fz_resnet_50_fz_anonymous6295_result.txt
CUDA_VISIBLE_DEVICES=3 python3 train.py --base_model deeplabv3plus_resnet50 --test 6 --resume /home/anonymous6295/scratch/models/deeplabv3/best_deeplabv3plus_resnet50_voc_os16.pth  --backbone resnet --lr 0.01 --workers 6 --batch_size 12  --checkname deeplab-resnet --eval-interval 1 --dataset cityscapes --save-interval 1  --densencloss 1e-6  --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 >fz_resnet_50_full_anonymous6295_result.txt
