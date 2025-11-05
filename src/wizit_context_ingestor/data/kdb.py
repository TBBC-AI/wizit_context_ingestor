from enum import Enum
from typing import Literal


class KdbServices(str, Enum):
    REDIS = "redis"
    CHROMA = "chroma"
    PG = "pg"


kdb_services = Literal[
    KdbServices.REDIS.value, KdbServices.CHROMA.value, KdbServices.PG.value
]
