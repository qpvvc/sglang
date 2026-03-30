TP_SIZE=8
MODEL_PATH=/mnt/seed-program-nas/001688/models/DeepSeek-V2-Lite
python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --trust-remote-code \
    --mem-fraction-static 0.76 \
    --host 0.0.0.0 \
    --port 30000 \
    --attention-backend flashinfer \
    --disable-cuda-graph \
    --tp-size $TP_SIZE \