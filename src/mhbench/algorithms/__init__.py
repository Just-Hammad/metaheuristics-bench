from .base import Algorithm
from .classic import RandomSearch, DE, PSO, GA
from .cmaes import CMAES

REGISTRY = {
    "random": RandomSearch,
    "de": DE,
    "pso": PSO,
    "ga": GA,
    "cmaes": CMAES,
}


def build(spec):
    """spec: {'name': 'de', 'params': {...}} -> Algorithm instance."""
    if isinstance(spec, str):
        return REGISTRY[spec]()
    return REGISTRY[spec["name"]](**spec.get("params", {}))


__all__ = ["Algorithm", "RandomSearch", "DE", "PSO", "GA", "CMAES", "REGISTRY", "build"]
