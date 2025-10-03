from enum import Enum
from typing import Literal


class KdbServices(Enum):
    REDIS = "redis"
    CHROMA = "chroma"


kdb_services = Literal[KdbServices.REDIS.value, KdbServices.CHROMA.value]
