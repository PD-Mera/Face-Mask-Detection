CONFIG_FILE=./configs/damoyolo_tinynasL20_T.py
CKPT=../workdirs/damoyolo_tinynasL20_T/damoyolo_tinynasL20_T.pth

cd DAMO-YOLO

python -m torch.distributed.launch \
    --nproc_per_node=1 tools/eval.py -f $CONFIG_FILE --ckpt $CKPT