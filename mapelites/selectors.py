from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from mapelites.interfaces import IIndividualSelector

if TYPE_CHECKING:
    from mapelites.archive import GridArchive


class RandomIndividualSelector(IIndividualSelector):
    def __init__(self, archive: GridArchive) -> None:
        self._archive = archive

    def get(self) -> Any:
        return random.choice(self._archive.non_empty_cells).solution
