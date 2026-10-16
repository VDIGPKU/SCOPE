"""Source profile for deterministic offline execution."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np

from .recording import SUPPORTED_EXCEPTIONS

NUMPY_NAMES = [
    "array",
    "asarray",
    "zeros",
    "ones",
    "full",
    "eye",
    "arange",
    "linspace",
    "zeros_like",
    "ones_like",
    "full_like",
    "copy",
    "reshape",
    "transpose",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "dot",
    "cross",
    "matmul",
    "abs",
    "clip",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "arctan2",
    "sum",
    "mean",
    "min",
    "max",
    "all",
    "any",
    "allclose",
    "isclose",
    "isfinite",
    "isnan",
    "isinf",
    "argmin",
    "argmax",
    "argsort",
    "sort",
    "squeeze",
    "expand_dims",
    "where",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool_",
    "pi",
    "nan",
    "inf",
]
ARRAY_ATTRIBUTES = {
    "shape",
    "strides",
    "dtype",
    "ndim",
    "size",
    "itemsize",
    "nbytes",
    "T",
    "base",
    "real",
    "imag",
    "copy",
    "reshape",
    "transpose",
    "ravel",
    "flatten",
    "squeeze",
    "astype",
    "item",
    "tolist",
    "sum",
    "mean",
    "min",
    "max",
    "all",
    "any",
    "argmin",
    "argmax",
    "clip",
    "round",
    "sort",
    "argsort",
    "fill",
}
METHODS = {
    list: {
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "index",
        "count",
        "reverse",
        "sort",
        "copy",
    },
    dict: {
        "get",
        "keys",
        "values",
        "items",
        "update",
        "pop",
        "popitem",
        "setdefault",
        "clear",
        "copy",
    },
    tuple: {"index", "count"},
    str: {
        "lower",
        "upper",
        "strip",
        "lstrip",
        "rstrip",
        "split",
        "rsplit",
        "join",
        "replace",
        "startswith",
        "endswith",
        "find",
        "index",
        "count",
        "isdigit",
        "isalpha",
    },
    bytes: {"decode", "count", "index"},
    bytearray: {"append", "extend", "pop", "clear", "reverse", "copy", "decode"},
}
FORBIDDEN_NAMES = {
    "open",
    "input",
    "print",
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "dir",
    "type",
    "id",
    "hash",
    "object",
    "super",
    "memoryview",
    "help",
    "breakpoint",
    "exit",
    "quit",
    "BaseException",
    "set",
    "frozenset",
}
ALLOWED_NODES = {
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Expr,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Try,
    ast.ExceptHandler,
    ast.Raise,
    ast.Assert,
    ast.Delete,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.Starred,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.MatMult,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.ListComp,
    ast.DictComp,
    ast.comprehension,
}


class ProfileError(ValueError):
    def __init__(self, reason, line=None):
        super().__init__(reason)
        self.reason, self.line = reason, line


class _Attributes(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_scope_attr", ctx=ast.Load()),
                    args=[node.value, ast.Constant(node.attr)],
                    keywords=[],
                ),
                node,
            )
        return node


def compile_source(source, entrypoint):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ProfileError("invalid_source", getattr(error, "lineno", None)) from error
    nodes = list(ast.walk(tree))
    if len(nodes) > 20000:
        raise ProfileError("source_complexity_limit")
    # Only statically check the injected facade when its name is unshadowed.
    # Aliases and locally bound values retain the runtime attribute checks.
    numpy_shadowed = any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "np"
        or isinstance(node, ast.arg) and node.arg == "np"
        or isinstance(node, ast.FunctionDef) and node.name == "np"
        or isinstance(node, ast.ExceptHandler) and node.name == "np"
        for node in nodes
    )
    for statement in tree.body:
        if not (
            isinstance(statement, ast.FunctionDef)
            or isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) is str
        ):
            raise ProfileError("module_statements_unsupported", statement.lineno)
    if sum(isinstance(n, ast.FunctionDef) and n.name == entrypoint for n in tree.body) != 1:
        raise ProfileError("missing_or_duplicate_entrypoint")
    for node in nodes:
        if type(node) not in ALLOWED_NODES:
            raise ProfileError("unsupported_syntax", getattr(node, "lineno", None))
        names = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        if isinstance(node, ast.arg):
            names.append(node.arg)
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
            if (
                node.decorator_list
                or node.returns
                or node.args.defaults
                or any(node.args.kw_defaults)
            ):
                raise ProfileError("function_metadata_unsupported", node.lineno)
        if isinstance(node, ast.arg) and node.annotation:
            raise ProfileError("annotations_unsupported", node.lineno)
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                raise ProfileError("bare_except_unsupported", node.lineno)
            if node.name:
                names.append(node.name)
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
            if (
                not numpy_shadowed
                and isinstance(node.value, ast.Name)
                and node.value.id == "np"
                and node.attr not in NUMPY_NAMES
            ):
                raise ProfileError("unsupported_numpy_attribute", node.lineno)
        if any(name.startswith("_") or name in FORBIDDEN_NAMES for name in names):
            raise ProfileError("forbidden_name_or_attribute", getattr(node, "lineno", None))
        if isinstance(node, ast.Raise) and node.cause is not None:
            raise ProfileError("exception_chaining_unsupported", node.lineno)
    tree = ast.fix_missing_locations(_Attributes().visit(tree))
    return compile(tree, "<scope-candidate>", "exec")


def execution_globals(api, read, inputs, public, unavailable):
    """Only data and tape endpoints are injected; no real API callbacks."""
    numpy = SimpleNamespace(**{name: getattr(np, name) for name in NUMPY_NAMES})

    def attribute(value, name):
        kind = type(value)
        allowed = False
        if kind is SimpleNamespace:
            allowed = name in vars(value) and not name.startswith("_")
        elif kind is np.ndarray:
            allowed = name in ARRAY_ATTRIBUTES
        elif isinstance(value, np.generic):
            allowed = name in {"item", "tolist", "dtype", "shape", "real", "imag"}
        elif isinstance(value, np.dtype):
            allowed = name in {"str", "name", "kind", "itemsize", "byteorder"}
        elif kind in METHODS:
            allowed = name in METHODS[kind]
        elif kind in SUPPORTED_EXCEPTIONS:
            allowed = name == "args"
        if not allowed:
            unavailable("unsupported_attribute", path=f"attribute.{name}")
        result = getattr(value, name)
        if isinstance(value, (np.ndarray, np.generic)) and callable(result):
            # NumPy's C methods can consult the calling frame's __import__ even
            # for ordinary reductions. Run supported methods in a host frame;
            # do not expose imports or change dtype objects in the np facade.
            def invoke(*args, **kwargs):
                return result(*args, **kwargs)

            return invoke
        return result

    builtins = dict(
        {
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "all": all,
            "any": any,
            "sorted": sorted,
            "reversed": reversed,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "int": int,
            "float": float,
            "bool": bool,
            "str": str,
            "bytes": bytes,
            "bytearray": bytearray,
            "round": round,
            "isinstance": isinstance,
        }.items()
    )
    builtins.update({cls.__name__: cls for cls in SUPPORTED_EXCEPTIONS})
    return {
        "__builtins__": builtins,
        "_scope_attr": attribute,
        "np": numpy,
        "api": api,
        "read": read,
        "inputs": inputs,
        "public": public,
    }
