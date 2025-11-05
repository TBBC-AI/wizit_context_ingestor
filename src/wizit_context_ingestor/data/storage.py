from enum import Enum
from typing import Literal


class StorageServices(str, Enum):
    S3 = "s3"
    LOCAL = "local"


storage_services = Literal[StorageServices.S3.value, StorageServices.LOCAL.value]
