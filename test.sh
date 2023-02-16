CONFIG_FILE=./configs/damoyolo_tinynasL20_T.py
ENGINE=../workdirs/damoyolo_tinynasL20_T/damoyolo_tinynasL20_T.pth
ENGINE_TYPE='torch'
CONFIDENCE=0.7
IMG_SIZE=640
DEVICE='cuda'
OUTPUT_DIR=../outputs/
IMG_TEST_PATH=../assets/test.jpg

cd DAMO-YOLO

python ./tools/demo.py -f $CONFIG_FILE \
                       --engine $ENGINE \
                       --engine_type $ENGINE_TYPE \
                       --conf $CONFIDENCE \
                       --infer_size $IMG_SIZE $IMG_SIZE \
                       --device $DEVICE \
                       --path $IMG_TEST_PATH \
                       --output_dir $OUTPUT_DIR