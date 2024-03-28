import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from mapelites.archive import GridArchive


@dataclass
class Fitness:
    fitness: float
    features_fitness: List[float]


class IFitnessFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, individual: Any) -> Fitness:
        raise NotImplementedError


class IComparator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, fitness1: float, fitness2: float) -> bool:
        raise NotImplementedError


class IIndividualSelector(abc.ABC):
    @abc.abstractmethod
    def get(self) -> Any:
        raise NotImplementedError


class IIndividualGenerator(abc.ABC):
    @abc.abstractmethod
    def create_individuals(self, count: int) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def get_mutated_individuals(
        self, individual_selector: IIndividualSelector, count: int
    ) -> Any:
        raise NotImplementedError
