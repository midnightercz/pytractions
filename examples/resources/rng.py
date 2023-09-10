import random

from pytraction.base import Base


class RNG(Base):
    seed: int = 0
    _rng: random.Random = None

    @property
    def rng(self):
        if not self._rng:
            self._rng = random.Random(self.seed)
        return self._rng

    def generate(self) -> int:
        return self.rng.randint(0, 100)
