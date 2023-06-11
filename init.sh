mkdir -p resources/datasets resources/checkpoints resources/checkpoints resources/db logs
pip3 install -r requirements.txt
python dump_data.py
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xvzf ./ngrok-v3-stable-linux-amd64.tgz && rm -rf ngrok-v3-stable-linux-amd64.tgz
./ngrok config add-authtoken 1v1HNV25PowMDNvS3rLi7wHcsc6_21L6shBzckabrHgScxeGA
echo "python3 train.py --batch_size 32 --seq_length 120 --hidden_size 128 --num_workers 2"
echo "tensorboard --logdir=logs"
echo "./ngrok http 6006"

# python3 train.py --batch_size 32 --seq_length 120 --hidden_size 128 --num_workers 16