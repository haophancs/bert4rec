import taskiq_fastapi
from taskiq import InMemoryBroker
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend

from bert4rec_service.settings import settings

result_backend = RedisAsyncResultBackend(
    redis_url=str(settings.redis_url.with_path("/1")),
)
broker = ListQueueBroker(
    str(settings.redis_url.with_path("/1")),
).with_result_backend(result_backend)

if settings.environment.lower() == "pytest":
    broker = InMemoryBroker()

taskiq_fastapi.init(
    broker,
    "bert4rec_service.web.application:get_app",
)
