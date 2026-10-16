"""Lossless state graphs for the explicitly supported SCOPE value domain.

This module provides the state graph data layer.
NumPy is imported only when this module is imported, keeping M1 standard-library
only. No pickle, dynamic imports from artifacts, or simulator APIs are used.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np

from .records import content_hash

FORMAT = "scope-state"
VERSION = 2
_EMPTY_TUPLE = ()
_HASH = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"o[1-9][0-9]*")


class StateUnavailable(ValueError):  # noqa: N818 - signals an unavailable certificate input
    """State cannot be represented/restored faithfully in the supported domain."""


class UnsupportedState(StateUnavailable):
    """An object type, layout, or cross-snapshot transition is unsupported."""


class InvalidState(StateUnavailable):
    """An encoded graph, payload, or reference is invalid or corrupted."""


class StateMismatch(StateUnavailable):
    """Live values/references do not match the graph requested for binding."""

    def __init__(self, message, *, path=None, expected=None, actual=None):
        super().__init__(message)
        self.path = path
        self.expected = expected
        self.actual = actual


def _state_difference(expected, actual, path):
    """First metadata difference, with bounded summaries (never array payloads)."""
    if type(expected) is dict and type(actual) is dict:
        if expected.keys() == actual.keys():
            for key in expected:
                if expected[key] != actual[key]:
                    return _state_difference(expected[key], actual[key], f"{path}.{key}")
        else:
            key = next(
                k
                for k in sorted(expected.keys() | actual.keys())
                if (k in expected) != (k in actual)
            )
            return (
                f"{path}.{key}",
                "present" if key in expected else "missing",
                "present" if key in actual else "missing",
            )
    if type(expected) is list and type(actual) is list:
        for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
            if left != right:
                return _state_difference(left, right, f"{path}[{index}]")
        return f"{path}.length", str(len(expected)), str(len(actual))
    return path, repr(expected)[:160], repr(actual)[:160]


@dataclass(frozen=True)
class CodecLimits:
    max_nodes: int = 10000
    max_payload_bytes: int = 256 * 1024 * 1024
    max_metadata_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if any(
            type(v) is not int or v < 1
            for v in (
                self.max_nodes,
                self.max_payload_bytes,
                self.max_metadata_bytes,
            )
        ):
            raise ValueError("codec limits must be positive integers")


DEFAULT_LIMITS = CodecLimits()


def _canonical_json(value: Any) -> str:
    # ASCII escapes preserve all Python strings, including unpaired surrogates.
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


@dataclass(frozen=True)
class PublicObjectSpec:
    """Opt-in field-only semantics for an exact trusted Python class.

    No constructors or attribute hooks run during restore. Only instance
    __dict__ fields listed here are supported; properties/slots and hidden fields
    are not silently captured. The caller must validate this contract for its API.
    """

    name: str
    cls: type
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_.-]*", self.name
        ):
            raise ValueError("invalid public object adapter name")
        if not isinstance(self.cls, type) or self.cls.__module__ in {"builtins", "numpy", "types"}:
            raise ValueError("public adapters require an explicit application class")
        if issubclass(
            self.cls, (list, dict, tuple, set, bytes, bytearray, np.ndarray, BaseException)
        ):
            raise ValueError("public adapters cannot hide builtin or array state")
        if any(base.__dict__.get("__slots__") for base in self.cls.__mro__):
            raise ValueError("slotted public objects require a separate explicit codec")
        if not isinstance(self.fields, tuple) or len(set(self.fields)) != len(self.fields):
            raise ValueError("fields must be a tuple of unique public names")
        if any(
            not isinstance(f, str) or not f.isidentifier() or f.startswith("_") for f in self.fields
        ):
            raise ValueError("fields must contain public attribute names")


def _registry(specs: tuple[PublicObjectSpec, ...]) -> dict[str, PublicObjectSpec]:
    names: dict[str, PublicObjectSpec] = {}
    classes: set[type] = set()
    for spec in specs:
        if not isinstance(spec, PublicObjectSpec) or spec.name in names or spec.cls in classes:
            raise ValueError("public object adapters must have unique names and classes")
        names[spec.name] = spec
        classes.add(spec.cls)
    return names


def _keys(value: Any, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        raise InvalidState(f"expected fields {sorted(expected)}")


def _integer(value: Any, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise InvalidState("invalid integer in state graph")
    return value


def _dtype(text: Any, *, scalar: bool = False) -> np.dtype:
    if type(text) is not str:
        raise InvalidState("dtype must be a string")
    try:
        dtype = np.dtype(text)
    except (TypeError, ValueError) as error:
        raise InvalidState("invalid NumPy dtype") from error
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.metadata is not None
        or dtype.kind not in "biufcSUmM"
        or (dtype.itemsize == 0 and not (scalar and dtype.kind in "SU"))
        or dtype.str != text
        or (dtype.kind == "f" and dtype.itemsize not in (2, 4, 8))
        or (dtype.kind == "c" and dtype.itemsize not in (8, 16))
    ):
        raise UnsupportedState(f"unsupported dtype: {text}")
    return dtype


def _scalar_from_bytes(dtype: np.dtype, data: bytes) -> np.generic:
    if dtype.kind == "S":
        scalar = np.bytes_(data[: dtype.itemsize])
    elif dtype.kind == "U":
        encoding = "utf-32-le" if dtype.str.startswith("<") else "utf-32-be"
        scalar = np.str_(data[: dtype.itemsize].decode(encoding, errors="surrogatepass"))
    else:
        scalar = np.frombuffer(data, dtype=dtype, count=1)[0]
    if scalar.dtype != dtype or scalar.tobytes() != data:
        raise InvalidState("scalar dtype/content cannot be reconstructed exactly")
    return scalar


def _fields(obj: object) -> dict[str, Any]:
    try:
        values = object.__getattribute__(obj, "__dict__")
    except AttributeError as error:
        raise UnsupportedState("public objects must have an instance dictionary") from error
    if type(values) is not dict:
        raise UnsupportedState("public object state is not a plain dictionary")
    return values


def _valid_key(value: Any) -> bool:
    if type(value) in (type(None), bool, int, str, bytes):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is tuple:
        return all(_valid_key(item) for item in value)
    return False


class StateSnapshot:
    """Immutable metadata and content-addressed binary payloads for one capture."""

    def __init__(self, metadata: dict[str, Any], blobs: dict[str, bytes]) -> None:
        try:
            self._json = _canonical_json(metadata)
        except (TypeError, ValueError, RecursionError) as error:
            raise InvalidState("metadata is not a finite JSON graph") from error
        if any(type(key) is not str or type(data) is not bytes for key, data in blobs.items()):
            raise InvalidState("blob storage must map hashes to immutable bytes")
        self._blobs = MappingProxyType(dict(blobs))

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def blobs(self) -> MappingProxyType:
        return self._blobs

    @property
    def sha256(self) -> str:
        return content_hash(self._json)

    def save(self, directory: Path, *, limits: CodecLimits = DEFAULT_LIMITS) -> Path:
        """Save into a new directory, publishing the manifest after all blobs."""
        directory = Path(directory)
        _validate(self, limits)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "blobs").mkdir()
        for digest, data in self._blobs.items():
            with (directory / "blobs" / f"{digest}.bin").open("xb") as stream:
                stream.write(data)
        path = directory / "state.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(
                {"sha256": self.sha256, "state": self.metadata}, stream, indent=2, ensure_ascii=True
            )
            stream.write("\n")
        return path

    @classmethod
    def load(cls, directory: Path, *, limits: CodecLimits = DEFAULT_LIMITS) -> StateSnapshot:
        directory = Path(directory)
        manifest = directory / "state.json"
        try:
            if manifest.is_symlink() or manifest.stat().st_size > limits.max_metadata_bytes:
                raise InvalidState("invalid or oversized state manifest")
            envelope = json.loads(
                manifest.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs
            )
            _keys(envelope, {"sha256", "state"})
            if envelope["sha256"] != content_hash(_canonical_json(envelope["state"])):
                raise InvalidState("metadata checksum mismatch")
            metadata = envelope["state"]
            descriptors = metadata["blobs"]
            if type(descriptors) is not dict or (directory / "blobs").is_symlink():
                raise InvalidState("invalid blob directory or descriptors")
            total = 0
            blobs = {}
            for digest, size in descriptors.items():
                if not _HASH.fullmatch(digest):
                    raise InvalidState("invalid blob digest")
                total += _integer(size, minimum=0)
                if total > limits.max_payload_bytes:
                    raise InvalidState("payload byte limit exceeded")
                path = directory / "blobs" / f"{digest}.bin"
                if path.is_symlink() or path.stat().st_size != size:
                    raise InvalidState("blob size mismatch or symlink")
                blobs[digest] = path.read_bytes()
            snapshot = cls(metadata, blobs)
            _validate(snapshot, limits)
            return snapshot
        except (OSError, KeyError, TypeError, ValueError, RecursionError) as error:
            if isinstance(error, StateUnavailable):
                raise
            raise InvalidState(f"cannot load state: {error}") from error


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidState("duplicate JSON key")
        result[key] = value
    return result


class StateEncoder:
    """Capture a serial execution's graph, retaining identities across captures.

    Previously observed reference objects are retained and re-captured even when
    absent from the next roots, because user code may still hold their aliases.
    Use a new encoder for each execution to release this retained state.
    """

    def __init__(
        self, *, specs: tuple[PublicObjectSpec, ...] = (), limits: CodecLimits = DEFAULT_LIMITS
    ) -> None:
        self._specs = _registry(specs)
        self._limits = limits
        self._session = uuid.uuid4().hex
        self._objects: dict[str, Any] = {}
        self._previous: dict[str, dict[str, Any]] = {}

    def capture(self, roots: dict[str, Any]) -> StateSnapshot:
        if type(roots) is not dict or any(type(name) is not str for name in roots):
            raise UnsupportedState("roots must be a plain dictionary with string names")
        capture = _Capture(self)
        try:
            encoded_roots = {name: capture.value(roots[name]) for name in sorted(roots)}
            for obj in tuple(self._objects.values()):
                capture.value(obj)
            metadata = {
                "format": FORMAT,
                "version": VERSION,
                "session": self._session,
                "roots": encoded_roots,
                "nodes": capture.nodes,
                "blobs": {digest: len(data) for digest, data in capture.blobs.items()},
            }
            snapshot = StateSnapshot(metadata, capture.blobs)
            _validate(snapshot, self._limits)
            _check_transitions(self._previous, capture.nodes)
        except RecursionError as error:
            raise UnsupportedState("state graph exceeds recursion depth") from error
        except (TypeError, ValueError, OverflowError) as error:
            if isinstance(error, StateUnavailable):
                raise
            raise UnsupportedState(f"cannot encode state: {error}") from error
        self._objects = capture.objects
        self._previous = capture.nodes
        return snapshot


class _Capture:
    def __init__(self, encoder: StateEncoder) -> None:
        self.encoder = encoder
        self.objects = dict(encoder._objects)
        self.ids = {id(obj): key for key, obj in self.objects.items()}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.blobs: dict[str, bytes] = {}

    def blob(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        if digest not in self.blobs:
            if (
                sum(map(len, self.blobs.values())) + len(data)
                > self.encoder._limits.max_payload_bytes
            ):
                raise UnsupportedState("payload byte limit exceeded")
            self.blobs[digest] = data
        return digest

    def value(self, obj: Any) -> dict[str, Any]:
        kind = type(obj)
        if obj is None:
            return {"type": "none"}
        if kind is bool:
            return {"type": "bool", "value": obj}
        if kind is int:
            return {"type": "int", "value": str(obj)}
        if kind is str:
            return {"type": "str", "value": obj}
        if kind is float:
            return {"type": "float", "bits": struct.pack(">d", obj).hex()}
        if kind is complex:
            return {"type": "complex", "bits": struct.pack(">dd", obj.real, obj.imag).hex()}
        object_id = self.ids.get(id(obj))
        if object_id is None:
            if len(self.objects) >= self.encoder._limits.max_nodes:
                raise UnsupportedState("object count limit exceeded")
            object_id = f"o{len(self.objects) + 1}"
            self.ids[id(obj)] = object_id
            self.objects[object_id] = obj
        if object_id in self.nodes:
            return {"ref": object_id}
        self.nodes[object_id] = {}  # Reserve before descending into cycles.
        if kind in (list, tuple):
            node = {"kind": kind.__name__, "items": [self.value(item) for item in obj]}
            if kind is tuple:
                node["canonical_empty"] = obj is _EMPTY_TUPLE
        elif kind is dict:
            if any(not _valid_key(key) for key in obj):
                raise UnsupportedState(
                    "dictionary keys must be supported immutable values (no NaN)"
                )
            node = {
                "kind": "dict",
                "items": [[self.value(key), self.value(value)] for key, value in obj.items()],
            }
        elif kind in (bytes, bytearray):
            node = {"kind": kind.__name__, "blob": self.blob(bytes(obj))}
        elif kind is np.ndarray:
            node = self.array(obj)
        elif isinstance(obj, np.generic) and kind is obj.dtype.type:
            dtype = _dtype(obj.dtype.str, scalar=True)
            node = {"kind": "numpy_scalar", "dtype": dtype.str, "blob": self.blob(obj.tobytes())}
        elif kind is SimpleNamespace or any(
            kind is spec.cls for spec in self.encoder._specs.values()
        ):
            fields = _fields(obj)
            spec = next((s for s in self.encoder._specs.values() if kind is s.cls), None)
            if any(
                type(name) is not str or not name.isidentifier() or name.startswith("_")
                for name in fields
            ):
                raise UnsupportedState("object has non-public fields")
            if spec is not None and not set(fields).issubset(spec.fields):
                raise UnsupportedState("object has fields not covered by its public adapter")
            node = {
                "kind": "object",
                "adapter": spec.name if spec else None,
                "attributes": self.value(fields),
            }
        else:
            raise UnsupportedState(f"unsupported state type: {kind.__module__}.{kind.__qualname__}")
        self.nodes[object_id] = node
        return {"ref": object_id}

    def array(self, array: np.ndarray) -> dict[str, Any]:
        if array.dtype.metadata is not None or array.flags.writebackifcopy:
            raise UnsupportedState("dtype metadata and writeback-if-copy arrays are unsupported")
        dtype = _dtype(array.dtype.str)
        node = {
            "kind": "ndarray",
            "dtype": dtype.str,
            "shape": list(array.shape),
            "strides": list(array.strides),
            "writable": bool(array.flags.writeable),
            "aligned": bool(array.flags.aligned),
            "base": None,
            "offset": 0,
            "blob": None,
        }
        if array.base is None:
            if not array.flags.owndata or not (
                array.flags.c_contiguous or array.flags.f_contiguous
            ):
                raise UnsupportedState("array owner must own contiguous memory")
            if array.nbytes > self.encoder._limits.max_payload_bytes:
                raise UnsupportedState("array payload byte limit exceeded")
            node["blob"] = self.blob(array.tobytes(order="A"))
        else:
            base = array.base
            if type(base) not in (np.ndarray, bytes, bytearray):
                raise UnsupportedState("array base must be ndarray, bytes or bytearray")
            node["base"] = self.value(base)
            if type(base) is np.ndarray:
                start = base.__array_interface__["data"][0]
            else:
                start = np.frombuffer(base, dtype=np.uint8).__array_interface__["data"][0]
            node["offset"] = array.__array_interface__["data"][0] - start
        return node


def _check_transitions(previous: dict[str, Any], nodes: dict[str, Any]) -> None:
    if not set(previous).issubset(nodes):
        raise UnsupportedState("snapshot dropped previously retained objects; start a new restorer")
    for key, old in previous.items():
        new = nodes[key]
        if old["kind"] != new["kind"]:
            raise UnsupportedState("an object ID changed type")
        kind = old["kind"]
        if kind == "ndarray":
            fixed = ("dtype", "shape", "strides", "base", "offset")
            if any(old[field] != new[field] for field in fixed):
                raise UnsupportedState("in-place ndarray layout changes are unsupported")
        elif kind in ("tuple", "bytes", "numpy_scalar") and old != new:
            raise UnsupportedState("immutable object changed under the same ID")
        elif kind == "object" and old != new:
            raise UnsupportedState("public object adapter or attribute dictionary changed")


def _validate(snapshot: StateSnapshot, limits: CodecLimits) -> dict[str, Any]:
    if len(snapshot._json.encode("utf-8")) > limits.max_metadata_bytes:
        raise InvalidState("metadata byte limit exceeded")
    metadata = snapshot.metadata
    _keys(metadata, {"format", "version", "session", "roots", "nodes", "blobs"})
    if (
        metadata["format"] != FORMAT
        or type(metadata["version"]) is not int
        or metadata["version"] != VERSION
    ):
        raise InvalidState("unsupported state format/version")
    if type(metadata["session"]) is not str or not re.fullmatch(
        r"[0-9a-f]{32}", metadata["session"]
    ):
        raise InvalidState("invalid encoder session")
    nodes, roots, descriptors = metadata["nodes"], metadata["roots"], metadata["blobs"]
    if (
        type(nodes) is not dict
        or len(nodes) > limits.max_nodes
        or type(roots) is not dict
        or type(descriptors) is not dict
    ):
        raise InvalidState("invalid or oversized graph")
    if set(nodes) != {f"o{index}" for index in range(1, len(nodes) + 1)}:
        raise InvalidState("object IDs must form a complete retained sequence")
    if set(descriptors) != set(snapshot.blobs):
        raise InvalidState("blob descriptors and payloads differ")
    total = 0
    for digest, size in descriptors.items():
        if not _HASH.fullmatch(digest):
            raise InvalidState("invalid blob hash")
        total += _integer(size, minimum=0)
        data = snapshot.blobs[digest]
        if total > limits.max_payload_bytes or len(data) != size:
            raise InvalidState("blob size/limit mismatch")
        if hashlib.sha256(data).hexdigest() != digest:
            raise InvalidState("blob checksum mismatch")
    used_blobs: set[str] = set()

    def blob(digest: Any) -> int:
        if type(digest) is not str or digest not in descriptors:
            raise InvalidState("missing blob reference")
        used_blobs.add(digest)
        return descriptors[digest]

    def value(token: Any) -> None:
        if type(token) is not dict:
            raise InvalidState("value token must be an object")
        if "ref" in token:
            _keys(token, {"ref"})
            if type(token["ref"]) is not str or token["ref"] not in nodes:
                raise InvalidState("dangling object reference")
        else:
            _literal(token)

    for token in roots.values():
        value(token)
    allocation_bytes = 0
    has_canonical_empty = False
    for key, node in nodes.items():
        if not _ID.fullmatch(key) or type(node) is not dict:
            raise InvalidState("invalid object ID/node")
        kind = node.get("kind")
        if kind in ("list", "tuple", "dict"):
            _keys(
                node, {"kind", "items", "canonical_empty"} if kind == "tuple" else {"kind", "items"}
            )
            if type(node["items"]) is not list:
                raise InvalidState("container items must be a list")
            if kind == "tuple":
                if type(node["canonical_empty"]) is not bool:
                    raise InvalidState("invalid canonical empty tuple flag")
                if node["canonical_empty"]:
                    if node["items"] or has_canonical_empty:
                        raise InvalidState("canonical empty tuple must be empty and unique")
                    has_canonical_empty = True
            for item in node["items"]:
                if kind == "dict":
                    if type(item) is not list or len(item) != 2:
                        raise InvalidState("dict entries must be pairs")
                    value(item[0])
                    value(item[1])
                else:
                    value(item)
        elif kind in ("bytes", "bytearray", "numpy_scalar"):
            _keys(node, {"kind", "blob", "dtype"} if kind == "numpy_scalar" else {"kind", "blob"})
            size = blob(node["blob"])
            allocation_bytes += size
            if kind == "numpy_scalar":
                dtype = _dtype(node["dtype"], scalar=True)
                expected = dtype.itemsize or (1 if dtype.kind == "S" else 4)
                if expected != size:
                    raise InvalidState("NumPy scalar payload size mismatch")
                _scalar_from_bytes(dtype, snapshot.blobs[node["blob"]])
        elif kind == "object":
            _keys(node, {"kind", "adapter", "attributes"})
            if node["adapter"] is not None and type(node["adapter"]) is not str:
                raise InvalidState("invalid adapter name")
            value(node["attributes"])
            if set(node["attributes"]) != {"ref"}:
                raise InvalidState("public attributes must reference a dictionary")
        elif kind == "ndarray":
            _keys(
                node,
                {
                    "kind",
                    "dtype",
                    "shape",
                    "strides",
                    "writable",
                    "aligned",
                    "base",
                    "offset",
                    "blob",
                },
            )
            dtype = _dtype(node["dtype"])
            shape, strides = node["shape"], node["strides"]
            if (
                type(shape) is not list
                or type(strides) is not list
                or len(shape) != len(strides)
                or len(shape) > 32
            ):
                raise InvalidState("invalid array dimensions")
            for size in shape:
                _integer(size, minimum=0)
            for stride in strides:
                _integer(stride)
            offset = _integer(node["offset"], minimum=0)
            if type(node["writable"]) is not bool or type(node["aligned"]) is not bool:
                raise InvalidState("invalid array writable/aligned flag")
            if math.prod(shape) * dtype.itemsize > limits.max_payload_bytes:
                raise InvalidState("logical array size exceeds limit")
            if node["base"] is None:
                if offset != 0 or blob(node["blob"]) != math.prod(shape) * dtype.itemsize:
                    raise InvalidState("invalid array owner payload")
                allocation_bytes += descriptors[node["blob"]]
            else:
                value(node["base"])
                if set(node["base"]) != {"ref"} or node["blob"] is not None:
                    raise InvalidState("invalid array base")
        else:
            raise UnsupportedState(f"unsupported node kind: {kind}")
        if allocation_bytes > limits.max_payload_bytes:
            raise InvalidState("decoded backing storage byte limit exceeded")
    if used_blobs != set(descriptors):
        raise InvalidState("unreferenced binary payload")
    return metadata


def _literal(token: dict[str, Any]) -> Any:
    kind = token.get("type")
    if kind == "none":
        _keys(token, {"type"})
        return None
    if kind in ("bool", "int", "str"):
        _keys(token, {"type", "value"})
        value = token["value"]
        if kind == "bool" and type(value) is bool:
            return value
        if kind == "str" and type(value) is str:
            return value
        if kind == "int" and type(value) is str:
            try:
                decoded = int(value)
                if str(decoded) == value:
                    return decoded
            except ValueError:
                pass
        raise InvalidState("invalid scalar value")
    if kind in ("float", "complex"):
        _keys(token, {"type", "bits"})
        bits = token["bits"]
        expected = 16 if kind == "float" else 32
        if type(bits) is not str or not re.fullmatch(f"[0-9a-f]{{{expected}}}", bits):
            raise InvalidState("invalid floating-point bits")
        unpacked = struct.unpack(">d" if kind == "float" else ">dd", bytes.fromhex(bits))
        return unpacked[0] if kind == "float" else complex(*unpacked)
    raise InvalidState("unsupported scalar token")


class StateRestorer:
    """Restore successive snapshots while keeping already-returned objects alive.

    Invalid graphs, payloads and unsupported layout transitions are preflighted
    before existing objects are changed. An unexpected failure during application
    poisons the restorer: its partially changed objects must not be used to certify
    anything. Use a new restorer for a different encoder session.
    """

    def __init__(
        self, *, specs: tuple[PublicObjectSpec, ...] = (), limits: CodecLimits = DEFAULT_LIMITS
    ) -> None:
        self._specs = _registry(specs)
        self._limits = limits
        self._session: str | None = None
        self._objects: dict[str, Any] = {}
        self._previous: dict[str, Any] = {}
        self._failed = False

    def bind(self, snapshot: StateSnapshot, roots: dict[str, Any]) -> None:
        """Bind an exactly matching live graph without changing its contents.

        This attaches program-created arguments to recorded IDs before a later
        restore applies API mutations. Matching is intentionally strict, including
        alias structure, dictionary order, dtype, layout, flags and scalar bits.
        Previously bound objects cannot be silently replaced by equal copies.
        """
        if self._failed:
            raise StateUnavailable("restorer was invalidated by a previous application failure")
        metadata = _validate(snapshot, self._limits)
        if self._session is not None and metadata["session"] != self._session:
            raise UnsupportedState("cannot mix encoder sessions")
        _check_transitions(self._previous, metadata["nodes"])
        verifier = StateEncoder(specs=tuple(self._specs.values()), limits=self._limits)
        verifier._objects = {
            key: self._objects[key]
            for key in sorted(self._objects, key=lambda value: int(value[1:]))
        }
        actual = verifier.capture(roots).metadata
        for field in ("roots", "nodes", "blobs"):
            if actual[field] != metadata[field]:
                path, expected, observed = _state_difference(
                    metadata[field], actual[field], f"$.{field}"
                )
                raise StateMismatch(
                    f"live {field} do not match the recorded state",
                    path=path,
                    expected=expected,
                    actual=observed,
                )
        self._session = metadata["session"]
        self._previous = metadata["nodes"]
        self._objects = verifier._objects

    def restore(self, snapshot: StateSnapshot) -> dict[str, Any]:
        if self._failed:
            raise StateUnavailable("restorer was invalidated by a previous application failure")
        try:
            metadata = _validate(snapshot, self._limits)
            if self._session is not None and metadata["session"] != self._session:
                raise UnsupportedState("cannot mix encoder sessions")
            _check_transitions(self._previous, metadata["nodes"])
            # Materialize a detached graph to validate keys, buffers and cycles
            # without touching values held by the candidate program.
            shadow = _Build(snapshot, metadata, self._specs, {})
            shadow.prepare()
            shadow.apply()
            plan = _Build(snapshot, metadata, self._specs, self._objects)
            plan.prepare()
        except StateUnavailable:
            raise
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
            raise InvalidState(f"cannot prepare restoration: {error}") from error
        try:
            plan.apply()
        except Exception as error:
            self._failed = True
            raise StateUnavailable("state application failed; discard this restorer") from error
        self._session = metadata["session"]
        self._previous = metadata["nodes"]
        self._objects = plan.objects
        return {name: plan.value(token) for name, token in metadata["roots"].items()}


def _check_bounds(node: dict[str, Any], available: int) -> None:
    shape, strides, offset = node["shape"], node["strides"], node["offset"]
    itemsize = _dtype(node["dtype"]).itemsize
    if any(size == 0 for size in shape):
        if not 0 <= offset <= available:
            raise InvalidState("empty array offset outside its buffer")
        return
    low = offset + sum(
        min(0, (size - 1) * stride) for size, stride in zip(shape, strides, strict=True)
    )
    high = (
        offset
        + sum(max(0, (size - 1) * stride) for size, stride in zip(shape, strides, strict=True))
        + itemsize
    )
    if low < 0 or high > available:
        raise InvalidState("array view exceeds its backing buffer")


class _Build:
    """Precompute all references and mutations before applying to live objects."""

    def __init__(
        self,
        snapshot: StateSnapshot,
        metadata: dict[str, Any],
        specs: dict[str, PublicObjectSpec],
        existing: dict[str, Any],
    ) -> None:
        self.snapshot = snapshot
        self.nodes = metadata["nodes"]
        self.specs = specs
        self.existing = existing
        self.objects = dict(existing)
        self.visiting: set[str] = set()
        self.prepared: set[str] = set()
        self.contents: dict[str, Any] = {}

    def prepare(self) -> None:
        # Reserve mutable containers before resolving cyclic edges or tuples.
        for key, node in self.nodes.items():
            kind = node["kind"]
            if kind in ("list", "dict"):
                expected = list if kind == "list" else dict
                if key in self.objects and type(self.objects[key]) is not expected:
                    raise UnsupportedState("live container type changed")
                self.objects.setdefault(key, expected())
            elif kind == "object":
                tag = node["adapter"]
                if tag is not None and tag not in self.specs:
                    raise UnsupportedState(f"missing explicit public object adapter: {tag}")
                expected = SimpleNamespace if tag is None else self.specs[tag].cls
                if key in self.objects:
                    if type(self.objects[key]) is not expected:
                        raise UnsupportedState("live public object type changed")
                else:
                    self.objects[key] = (
                        SimpleNamespace() if tag is None else object.__new__(expected)
                    )
        # SimpleNamespace owns a non-replaceable __dict__. Associate that actual
        # dictionary with the graph ID, preserving aliases obtained via vars().
        namespace_owners: set[str] = set()
        for key, node in self.nodes.items():
            if node["kind"] != "object":
                continue
            attribute_id = node["attributes"]["ref"]
            if self.nodes[attribute_id]["kind"] != "dict":
                raise InvalidState("public object attributes must be a dictionary")
            if node["adapter"] is None:
                if attribute_id in namespace_owners:
                    raise InvalidState("two namespaces cannot own the same attribute dictionary")
                namespace_owners.add(attribute_id)
                actual = _fields(self.objects[key])
                if attribute_id in self.existing and self.objects[attribute_id] is not actual:
                    raise UnsupportedState("namespace attribute identity changed")
                self.objects[attribute_id] = actual
        for key, node in self.nodes.items():
            if node["kind"] == "object":
                attributes = self.objects[node["attributes"]["ref"]]
                if key in self.existing:
                    if _fields(self.objects[key]) is not attributes:
                        raise UnsupportedState("live public attribute dictionary changed")
                elif node["adapter"] is not None:
                    object.__setattr__(self.objects[key], "__dict__", attributes)
        for key in self.nodes:
            self.make(key)
        for key, node in self.nodes.items():
            if node["kind"] == "list":
                self.contents[key] = [self.value(token) for token in node["items"]]
            elif node["kind"] == "dict":
                entries = [(self.value(k), self.value(v)) for k, v in node["items"]]
                if any(not _valid_key(k) for k, _ in entries):
                    raise UnsupportedState("unsupported dictionary key")
                if len({k for k, _ in entries}) != len(entries):
                    raise InvalidState("duplicate/equal dictionary keys")
                self.contents[key] = entries
        for node in self.nodes.values():
            if node["kind"] == "object":
                fields = [name for name, _ in self.contents[node["attributes"]["ref"]]]
                if any(
                    type(name) is not str or not name.isidentifier() or name.startswith("_")
                    for name in fields
                ):
                    raise UnsupportedState("object contains non-public fields")
                if node["adapter"] is not None and not set(fields).issubset(
                    self.specs[node["adapter"]].fields
                ):
                    raise UnsupportedState("object fields exceed its explicit adapter")

    def value(self, token: dict[str, Any]) -> Any:
        return self.make(token["ref"]) if "ref" in token else _literal(token)

    def make(self, key: str) -> Any:
        if key in self.prepared:
            return self.objects[key]
        if key in self.visiting:
            raise InvalidState("cycle through immutable objects or array bases")
        node = self.nodes[key]
        kind = node["kind"]
        if kind in ("list", "dict", "object"):
            self.prepared.add(key)
            return self.objects[key]
        self.visiting.add(key)
        if kind == "tuple":
            values = (
                _EMPTY_TUPLE
                if node["canonical_empty"]
                else tuple(self.value(token) for token in node["items"])
            )
            if not node["canonical_empty"] and values is _EMPTY_TUPLE:
                raise UnsupportedState("runtime cannot reconstruct a noncanonical empty tuple")
            if key not in self.objects:
                self.objects[key] = values
            elif type(self.objects[key]) is not tuple or len(self.objects[key]) != len(values):
                raise UnsupportedState("live tuple changed")
            elif (self.objects[key] is _EMPTY_TUPLE) != node["canonical_empty"]:
                raise UnsupportedState("live empty tuple identity changed")
        elif kind in ("bytes", "bytearray", "numpy_scalar"):
            data = self.snapshot.blobs[node["blob"]]
            if key not in self.objects:
                if kind == "bytes":
                    self.objects[key] = memoryview(data).tobytes()
                elif kind == "bytearray":
                    self.objects[key] = bytearray(data)
                else:
                    self.objects[key] = _scalar_from_bytes(_dtype(node["dtype"], scalar=True), data)
            if kind == "bytearray" and (
                type(self.objects[key]) is not bytearray or len(self.objects[key]) != len(data)
            ):
                raise UnsupportedState("in-place bytearray resizing is unsupported")
        elif kind == "ndarray":
            self.array(key, node)
        else:
            raise InvalidState("unknown object kind")
        self.visiting.remove(key)
        self.prepared.add(key)
        return self.objects[key]

    def array(self, key: str, node: dict[str, Any]) -> None:
        dtype, shape, strides = _dtype(node["dtype"]), tuple(node["shape"]), tuple(node["strides"])
        base = self.value(node["base"]) if node["base"] is not None else None
        if base is None:
            available = len(self.snapshot.blobs[node["blob"]])
        elif type(base) is np.ndarray:
            if not (base.flags.c_contiguous or base.flags.f_contiguous):
                raise UnsupportedState("array base must be contiguous")
            available = base.nbytes
        elif type(base) in (bytes, bytearray):
            available = len(base)
        else:
            raise InvalidState("invalid array backing object")
        _check_bounds(node, available)
        if key not in self.objects:
            if base is None:
                array = np.empty(shape, dtype=dtype)
                array.strides = strides
                if not (array.flags.c_contiguous or array.flags.f_contiguous):
                    raise UnsupportedState("array owner must be contiguous")
            else:
                array = np.ndarray(
                    shape, dtype=dtype, buffer=base, offset=node["offset"], strides=strides
                )
                if array.base is not base:
                    raise UnsupportedState("NumPy constructor changes the recorded base identity")
            self.objects[key] = array
        array = self.objects[key]
        if (
            type(array) is not np.ndarray
            or array.dtype.str != node["dtype"]
            or array.dtype.metadata is not None
            or array.flags.writebackifcopy
            or array.shape != shape
            or array.strides != strides
            or array.base is not base
        ):
            raise UnsupportedState("live array layout/base changed")
        if base is not None:
            start = (
                base.__array_interface__["data"][0]
                if type(base) is np.ndarray
                else np.frombuffer(base, dtype=np.uint8).__array_interface__["data"][0]
            )
            if array.__array_interface__["data"][0] - start != node["offset"]:
                raise UnsupportedState("live array byte offset changed")
        ultimate = base
        while type(ultimate) is np.ndarray:
            ultimate = ultimate.base
        if type(ultimate) is bytes and node["writable"]:
            raise InvalidState("array backed by immutable bytes cannot be writable")

    def apply(self) -> None:
        # Enable owner memory first. View flags are independent: a writable view
        # can legitimately survive its owner becoming read-only afterwards.
        for key, node in self.nodes.items():
            if node["kind"] == "ndarray" and node["base"] is None:
                array = self.objects[key]
                array.setflags(write=True)
                raw = array.ravel(order="K").view(np.uint8)
                raw[:] = np.frombuffer(self.snapshot.blobs[node["blob"]], dtype=np.uint8)
            elif node["kind"] == "bytearray":
                self.objects[key][:] = self.snapshot.blobs[node["blob"]]
        # All arrays are prepared recursively (bases first), unlike JSON key order.
        arrays = [
            key
            for key in self.objects
            if key in self.nodes and self.nodes[key]["kind"] == "ndarray"
        ]
        for key in arrays:
            self.objects[key].setflags(align=self.nodes[key]["aligned"])
            if self.nodes[key]["writable"]:
                self.objects[key].setflags(write=True)
        for key in arrays:
            if not self.nodes[key]["writable"]:
                self.objects[key].setflags(write=False)
        for key, contents in self.contents.items():
            obj = self.objects[key]
            if self.nodes[key]["kind"] == "list":
                obj[:] = contents
            else:
                obj.clear()
                obj.update(contents)
