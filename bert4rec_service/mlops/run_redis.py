import os
from dotenv import load_dotenv
import subprocess

load_dotenv()

redis_port = os.getenv('REDIS_PORT', '6379')
redis_host = os.getenv('REDIS_HOST', 'localhost')

redis_server_command = f'redis-server --port {redis_port} --bind {redis_host}'
subprocess.run(redis_server_command, shell=True)
