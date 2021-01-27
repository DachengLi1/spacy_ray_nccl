from typing import Dict, Set, Iterable, Any, Optional, cast
from collections import Counter
from thinc.types import FloatsXd
from thinc.api import Optimizer
from spacy.util import logger
from .util import make_key, KeyT
import  ray.util.collective as collective
import numpy
import cupy
from cupy.cuda import Device

class RayPeerProxy:
    """Proxy for workers where each worker owns some of the parameters. For
    parameters they don't own, workers will pull parameters and push gradients.
    For parameters they do own, they pull gradients, make the update, and push
    parameters.
    """

    ray: Any
    optimizer: Optimizer
    grads_per_update: int
    peers: Dict
    other_workers: Set
    _params: Dict[KeyT, FloatsXd]
    _grads: Dict[KeyT, Optional[FloatsXd]]
    _versions: Dict[KeyT, int]
    _owned_keys: Set[KeyT]
    _grad_counts: Dict[KeyT, int]

    def __init__(
        self,
        peers: Dict[KeyT, Any],
        optimizer,
        keys: Iterable[KeyT],
        *,
        grads_per_update: int = 2,
        ray=None,
        rank=None,
        world_size=None
    ):
        if ray is None:
            import ray  # type: ignore
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        self.optimizer = optimizer
        self.grads_per_update = grads_per_update
        self.peers = dict(peers)
        self._owned_keys = set(keys)
        self.other_workers = set()
        for _, peer in self.peers.items():
            self.other_workers.add(peer)
        self.num_workers = world_size
        self.rank = rank
        assert self.rank is not None and self.num_workers is not None
        logger.info("There are %s workers.", self.num_workers)
        self._params = {}
        self._versions = Counter()
        self._next_params = {}
        self._grads = {}
        self._grad_counts = Counter()

    def check_version(self, key: KeyT, version: int) -> Optional[bool]:
        if key not in self._versions:
            return None
        elif self._versions[key] != version:
            return False
        else:
            return True

    def set_param(self, id, name, value: FloatsXd) -> None:
        """Set a parameter to the connection."""
        key = make_key(id, name)
        if key in self._owned_keys or key not in self._params:
            self._params[key] = value
            self._versions[key] += 1
            self._grads[key] = None
            self._grad_counts[key] = 0

    def send_param(self, key):
        version = self._versions[key]
        with Device(0):
            comm_value = cupy.asarray(self._params[key].copy())
        # allreduce the param value and average.
        #collective.allreduce(comm_value, "default")
        #comm_value /= self.num_workers
        # Place the new values to the original device.
        if isinstance(self._params[key], numpy.ndarray):
            self._params[key] = comm_value.get()
        else:
            self._params[key] = comm_value
        self._versions[key] += 1
        self._grads[key] = None
        self._grad_counts[key] = 0

    def receive_param(self, key, version, value: FloatsXd) -> None:
        """Let the connection push a parameter to us."""
        # We have to store this in a separate place, to make sure we don't
        # fetch the wrong version when we submit the gradient. For instance,
        # imagine if we received the param in between the forward and backward
        # pass. If we set the version to this one, we'd calculate a gradient
        # on the basis of the old param, but think we had a new version.
        self._next_params[key] = (version, value)

    def get_param(self, id, name) -> FloatsXd:
        key = make_key(id, name)
        self._maybe_update_param(key)
        return self._params[key]

    def set_grad(self, id, name, value: FloatsXd) -> None:
        """Set a gradient to the connection."""
        key = make_key(id, name)
        if key in self._owned_keys:
            self._grads[key] = value
            self._grad_counts[key] = 1

    def inc_grad(self, id, name, value: FloatsXd) -> None:
        """Increment a gradient to the connection."""
        key = make_key(id, name)
        self._grad_counts[key] += 1
        
        with Device(0):
            comm_value = cupy.asarray(value.copy() / 1)
        collective.allreduce(comm_value, "default")
        if isinstance(self._params[key], numpy.ndarray):
            value = comm_value.get()
        else:
            value = comm_value

        if self._grads.get(key) is None:
            self._grads[key] = value.copy()
        else:
            self._grads[key] += value

    def _maybe_update_param(self, key: KeyT) -> bool:
        if key in self._next_params:
            version, value = self._next_params.pop(key)
            self._params[key] = value
            self._versions[key] = version
            self._grad_counts[key] = 0
            self._grads[key] = None
            return True
        elif key not in self._owned_keys:
            return False
        elif self._grad_counts[key] < self.grads_per_update:
            return False
        elif self._grads.get(key) is None:
            return False
        else:
            grad = cast(FloatsXd, self._grads[key])
            self._versions[key] += 1
            param, _ = self.optimizer(key, self._params[key], grad)
            self._params[key] = param
            self._grads[key] = None
            self._grad_counts[key] = 0
            self.send_param(key)
            return True
