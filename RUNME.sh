python3 -m venv venv
source venv/bin/activate
mkdir -p resources/datasets resources/checkpoints resources/checkpoints resources/db logs
pip3 install -r requirements.txt
python3 run_dump_data.py
gdown https://drive.google.com/u/0/uc?id=1gf4_zpHd4H-ZH625TvcSEEs95idxYGOa
mv bert4rec_ml-25m_best.ckpt resources/checkpoints/

echo "ATTENTION!!!!!"
echo "Run the following commands in different terminal tabs:"
echo "python3 run_redis.py"
echo "celery -A run_api.celery worker --loglevel=info --pool=solo"
echo "uvicorn run_api:app --reload"
