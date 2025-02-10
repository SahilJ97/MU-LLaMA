import os
import redis
from redis_layer import RedisQueue
import dotenv


dotenv.load_dotenv()

# Configure Redis client and queues
ENVIRONMENT = os.environ.get('ENVIRONMENT')
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
redis = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,  # Automatically decode responses to strings instead of bytes
    socket_timeout=5,  # Add timeout to prevent hanging
    retry_on_timeout=True,  # Automatically retry on timeout
)
audio_redis_q = RedisQueue(redis, f'{ENVIRONMENT}_audio')
