# from the tutorial below
# https://tech.preferred.jp/en/blog/working-with-configuration-in-python/

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Backend(Enum):
    MYSQL = 1
    POSTGRES = 2
    SQLITE = 3


@dataclass
class Configuration:
    version: str
    name: str
    port: int
    backend: Backend
    bin_path: Path
    dataset_path: Path
    lr: float
    batch_size: int
    epochs:int


CONFIG = Configuration(
    version='1.1',
    name='David',
    port=8080,
    backend=Backend.MYSQL,
    bin_path=Path(r'G:\bin_pointnet3'),
    dataset_path=Path(r'F:\dataset\ModelNet10'),
    lr=1e-3,
    batch_size=32,
    epochs=15,
)

CONFIG.bin_path.mkdir(parents=True, exist_ok=True)
