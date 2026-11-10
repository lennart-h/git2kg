from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Self,
    Set,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

DP = TypeVar("DP", bound="Datapoint")


@dataclass
class Convertable(ABC):  # noqa: B024
    def to_dict(self) -> dict:
        """
        Converts the convertable object to a dictionary. Hereby, nested values are flattened to singular values, prefixed by the nested field name. Lists of nested values are flattened to lists of singular values, prefixed by the nested field name.
        """
        base_dict = asdict(self)
        for field_name, field_type in self.dtypes().items():
            if isinstance(field_type, type) and issubclass(field_type, NestedNodeInfo):
                unwrapped = unwrap_nested(
                    self.__getattribute__(field_name), field_name, field_type
                )
                base_dict.update(unwrapped)
                del base_dict[field_name]
            if get_origin(field_type) is list:
                list_args = get_args(field_type)
                if not list_args:
                    continue
                ninfo_type = list_args[0]
                if issubclass(ninfo_type, NestedNodeInfo):
                    unwrapped = unwrap_list_nested(
                        self.__getattribute__(field_name), field_name, ninfo_type
                    )
                    base_dict.update(unwrapped)
                    del base_dict[field_name]
        return base_dict

    @classmethod
    @lru_cache(maxsize=None)
    def dtypes(cls, flatten=False) -> dict[str, type]:
        """The fields of the datapoint and their types.

        Cached: resolving the annotations is far more expensive than anything
        done with the result, and converting a stored table asks for them once
        per row. The result is shared, so callers must not change it.
        """
        type_dict = get_type_hints(cls)
        unwrapped_types = {
            key: unwrap_optional(dtype) for key, dtype in type_dict.items()
        }
        if not flatten:
            return unwrapped_types

        result = {}
        for field_name, dtype in unwrapped_types.items():
            result.update(_flatten_type_entry(field_name, dtype))
        return result

    @classmethod
    def from_flat_dict(cls, flatfields: dict[str, Any]) -> Self:
        flatfields = dict(flatfields)
        nested_args: dict[str, Any] = {}
        for field, dtype in cls.dtypes().items():
            if isinstance(dtype, type) and issubclass(dtype, NestedNodeInfo):
                args = {
                    nf.name: flatfields.pop(f"{field}_{nf.name}", None)
                    for nf in fields(dtype)
                }
                nested_args[field] = dtype(**args)

                is_none = all(arg is None for arg in args.values())
                if is_none:
                    nested_args[field] = None
            origin_type = get_origin(dtype)
            if origin_type is list:
                list_type = get_args(dtype)[0]
                if issubclass(list_type, NestedNodeInfo):
                    # Find all columns related to the nested
                    list_args = {}
                    for nf in fields(list_type):
                        val = flatfields.pop(f"{field}_{nf.name}", origin_type())
                        val = origin_type() if val is None else val
                        list_args[nf.name] = val

                    length = max([len(_list) for _list in list_args.values()])
                    for key, list_arg in list_args.items():
                        if len(list_arg) == 0:
                            list_args[key] = origin_type([None for i in range(length)])
                    args_list = [
                        dict(zip(list_args.keys(), values, strict=True))
                        for values in zip(*list_args.values(), strict=True)
                    ]

                    nested_args[field] = origin_type(
                        [list_type(**arg_dict) for arg_dict in args_list]
                    )
        nested_args.update(flatfields)
        return cls(**nested_args)


@dataclass
class NestedNodeInfo(Convertable):
    @abstractmethod
    def get_id(self):
        pass


@dataclass
class ActorInfo(NestedNodeInfo):
    name: str
    email: Optional[str] = None
    github_id: Optional[int] = None
    type: Optional[str] = None
    id: Optional[int] = None

    # Column names used for actor_id/actor_github_id once ActorInfo instances
    # are disambiguated and merged into a DataFrame (e.g. actor_mapping).
    # Not type-annotated, so the dataclass machinery doesn't treat these as fields.
    ID_COLUMN = "actor_id"
    GITHUB_ID_COLUMN = "actor_github_id"
    TYPE_COLUMN = "actor_type"

    def __hash__(self):
        return hash((self.name, self.email, self.github_id, self.type, self.id))

    def __eq__(self, other):
        if not isinstance(other, ActorInfo):
            return NotImplemented
        return (self.name, self.email, self.github_id, self.type, self.id) == (
            other.name,
            other.email,
            other.github_id,
            other.type,
            other.id,
        )

    def get_id(self):
        return self.id


@dataclass
class IssueInfo(NestedNodeInfo):
    number: int

    def __hash__(self):
        return hash(self.number)

    def __eq__(self, other):
        if not isinstance(other, IssueInfo):
            return NotImplemented
        return self.number == other.number

    def get_id(self):
        return self.number


@dataclass
class CommitInfo(NestedNodeInfo):
    hash: str
    id: Optional[int] = None

    def get_id(self):
        return self.id


@dataclass
class FileInfo(NestedNodeInfo):
    commit_hash: str
    path: str
    id: Optional[int] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, FileInfo):
            return NotImplemented
        return self.id == other.id

    def get_id(self):
        return self.id


@dataclass
class Datapoint(Convertable):
    @abstractmethod
    def from_field(self) -> Any:
        """Returns the value of the field that should be used to determine the continuity of datapoints in mining. Passed as a lower bound to the mining algorithm."""
        pass

    @classmethod
    @abstractmethod
    def appearance_key(cls) -> Callable[["Datapoint"], Any]:
        """Returns an accessor for the time at which the entity described by
        this datapoint becomes observable, i.e. when it starts to exist. This
        is what places the datapoint on the timeline when slicing into
        snapshots.
        """
        pass

    @classmethod
    def mining_key(cls) -> Callable[["Datapoint"], Any]:
        """Returns an accessor for the timestamp mining resumes from, used to
        find the newest datapoint already mined.

        Defaults to `appearance_key`. Override where an entity can change after
        it appeared, so that mining picks up those changes: an issue is mined
        again whenever it was updated, not only when it was created.
        """
        return cls.appearance_key()


class DatapointList(Generic[DP]):
    def __init__(self, datapoints: list[DP], dp_type: type[DP] | None = None):
        if len(datapoints) == 0 and dp_type is None:
            raise ValueError("Specify dp_type when initializing an empty DatapointList")
        types = {type(dp) for dp in datapoints}
        if len(types) > 1:
            raise ValueError("All datapoints in the list must be of the same type")
        inferred_dp_type = None
        if len(types) > 0:
            inferred_dp_type = types.pop()
        if (dp_type or inferred_dp_type) != (inferred_dp_type or dp_type):
            raise ValueError(
                f"Specified datapoint type {dp_type.__class__.__name__} differs from actual datapoint type {inferred_dp_type.__class__.__name__}"
            )
        resolved_dp_type = dp_type or inferred_dp_type
        assert resolved_dp_type is not None
        self.datapoints = datapoints
        self.dp_type: type[DP] = resolved_dp_type

    def append(self, dp: DP):
        if type(dp) is not self.dp_type:
            raise ValueError(
                f"Datapoint type {type(dp)} does not match list type {self.dp_type}"
            )
        self.datapoints.append(dp)

    def extend(self, dps: "DatapointList[DP]"):
        for dp in dps:
            self.append(dp)

    def find_latest(self) -> DP:
        return max(self.datapoints, key=self.dp_type.mining_key())

    def __len__(self):
        return len(self.datapoints)

    def __iter__(self):
        return iter(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError("DatapointList can only joined with another DatapointList")
        if self.dp_type != other.dp_type:
            raise ValueError(
                f"Datapoint types of joined lists must be equal. Expected {self.dp_type}, but got {other.dp_type}"
            )
        dp_copy = list(self.datapoints)
        dp_copy.extend(other.datapoints)
        return DatapointList(dp_copy, self.dp_type)

    def to_dict_list(self) -> list[dict]:
        return [dp.to_dict() for dp in self.datapoints]


def unwrap_optional(t: type) -> type:
    if get_origin(t) is Union and len(get_args(t)) == 2 and type(None) in get_args(t):
        return next(a for a in get_args(t) if a is not type(None))
    return t


def _flatten_dataclass_fields(cls) -> dict[str, type]:
    """Flatten all fields of a NestedNodeInfo subclass into leaf_name -> type."""
    resolved = get_type_hints(cls)  # resolves forward refs / stringified annotations
    result = {}
    for f in fields(cls):
        field_type = unwrap_optional(resolved[f.name])
        result.update(_flatten_type_entry(f.name, field_type))
    return result


def _flatten_type_entry(key, dtype):
    dtype = unwrap_optional(dtype)
    origin = get_origin(dtype)

    # Case 1: List[...] / Set[...]
    if origin in (list, set):
        (elem_type,) = get_args(dtype)
        elem_type = unwrap_optional(elem_type)

        if isinstance(elem_type, type) and issubclass(elem_type, NestedNodeInfo):
            container = List if origin is list else Set
            leaf_entries = _flatten_dataclass_fields(elem_type)
            return {
                f"{key}_{leaf_key}": container[leaf_type]
                for leaf_key, leaf_type in leaf_entries.items()
            }
        # collection of primitives (e.g. Set[str]) — leave as-is
        return {key: dtype}

    # Case 2: a bare nested dataclass (not inside a collection)
    if isinstance(dtype, type) and issubclass(dtype, NestedNodeInfo):
        return {
            f"{key}_{leaf_key}": leaf_type
            for leaf_key, leaf_type in _flatten_dataclass_fields(dtype).items()
        }

    # Base case: primitive / opaque type
    return {key: dtype}


def unwrap_nested(
    nested_info: NestedNodeInfo,
    field_name,
    nested_cls: type[NestedNodeInfo],
) -> dict[str, Any]:
    if nested_info:
        nested_dict = asdict(nested_info)
    else:
        nested_dict = {f.name: None for f in fields(nested_cls)}

    unwrapped = {f"{field_name}_{k}": v for k, v in nested_dict.items()}
    return unwrapped


def unwrap_list_nested(
    nested_info_list: List[NestedNodeInfo],
    field_name: str,
    nested_cls: type[NestedNodeInfo],
):

    nested_dicts = []
    for ninfo in nested_info_list:
        nested_dicts.append(asdict(ninfo))
    unwrapped = {}
    for nfield in map(lambda f: f.name, fields(nested_cls)):
        unwrapped[f"{field_name}_{nfield}"] = [ninfo[nfield] for ninfo in nested_dicts]
    return unwrapped
