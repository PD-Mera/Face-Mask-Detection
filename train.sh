CONFIG_FILE=./configs/damoyolo_tinynasL20_T.py


cp ./config/base.py ./DAMO-YOLO/damo/config/
cp ./config/paths_catalog.py ./DAMO-YOLO/damo/config/
cp ./config/damoyolo_tinynasL20_T.py ./DAMO-YOLO/configs/

cd DAMO-YOLO

python -m torch.distributed.launch --nproc_per_node=1 \
    tools/train.py -f $CONFIG_FILE \
                   --local_rank 0