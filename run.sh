if [ ! -e './logs' ]; then
    mkdir ./logs
fi
if [ ! -e './.cache' ]; then
    mkdir ./.cache
fi

time=$(date "+%Y%m%d-%H%M%S")
nohup python -u ./main.py 2>&1 > "./logs/training-$time.log" &  