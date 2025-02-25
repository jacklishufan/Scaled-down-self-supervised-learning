python main_moco_pretraining.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 128 --epochs 64058 \
  --input-size 56 \
  --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3 \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  /home/jacklishufan/ddssl/ckpt/02-23-2023/syn_videos_950.pth
