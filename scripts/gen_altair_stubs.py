"""Generate .pyi stubs for the Altair MotionView Python API (``hw.mview``).

Altair ships ``hw/python/hw`` as bytecode only (``.pyc``, no source, no stubs).
Pylance never indexes bytecode, so the modelling API we script MotionView with
gets no editor completion and every import shows as unresolved.

This walks the shipped ``.pyc`` files, unmarshals their code objects, and emits
skeleton stubs into ``typings/`` -- Pylance's default ``stubPath``. The stubs
carry classes, bases, method signatures, the ``Double``/``Reference``/...
properties declared in each class body, the docstrings (bytecode keeps those
unless compiled with -OO), and a link to each entity's page in Altair's online
help, which every entity class records in ``_helpFilePython``.

Parameter defaults and real types are *not* recoverable from bytecode, so every
parameter is emitted as optional and typed ``Any``. This is a completion aid,
not a type contract -- it will not catch a wrong or missing argument. The help
link on each class is the authority for that.

Must run under **CPython 3.10**: marshal is version specific and the shipped
bytecode is 3.10. Altair bundles a matching interpreter.

    & "C:\\Program Files\\Altair\\2026.1\\common\\python\\python3.10\\win64\\python.exe" scripts\\gen_altair_stubs.py

Re-run it after an Altair upgrade; ``typings/`` is disposable output.
"""

from __future__ import annotations

import argparse
import ast
import dis
import importlib.util
import inspect
import marshal
import shutil
import sys
import types
from pathlib import Path

DEFAULT_ALTAIR_ROOT = Path(r"C:\Program Files\Altair\2026.1")
# Package root inside the install: <root>/hwdesktop/hw/python/<PACKAGE>
PACKAGE_PARENT = Path("hwdesktop/hw/python")
PACKAGE = "hw"

# Entity classes record their own help topic, e.g. Point._helpFilePython is
# "topics/reference/mv/point_python_r.htm". Prefixing this base gives the live
# page, which carries the argument types and defaults the bytecode does not.
HELP_BASE = "https://help.altair.com/hwdesktop/hwx/"
HELP_ATTRIBUTES = ("_helpFilePython", "_helpFile")

CO_OPTIMIZED = 0x01
CO_VARARGS = 0x04
CO_VARKEYWORDS = 0x08

# MotionView declares entity attributes with these property descriptors in the
# class body (see hw/mview/mbd/Object.pyc). Map them to something useful.
PROPERTY_TYPES = {
    "Double": "float",
    "Int": "int",
    "Integer": "int",
    "Bool": "bool",
    "Boolean": "bool",
    "String": "str",
    "File": "str",
    "Enum": "str",
    "Filename": "str",
    "Reference": "Any",
    "Instance": "Any",
    "ObjectList": "Any",
    "Expression": "Any",
    "CachedExpression": "Any",
    "ExpressionList": "Any",
}

SKIPPED_CODE_NAMES = {"<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>", "<lambda>"}


class ParseError(Exception):
    pass


# --------------------------------------------------------------------------
# bytecode reading
# --------------------------------------------------------------------------


def load_code(path: Path) -> types.CodeType:
    """Unmarshal the module code object out of a .pyc file."""
    data = path.read_bytes()
    if len(data) < 16:
        raise ParseError("file too short to be a .pyc")
    if data[:4] != importlib.util.MAGIC_NUMBER:
        raise ParseError(
            f"bytecode magic {data[:4]!r} != interpreter magic "
            f"{importlib.util.MAGIC_NUMBER!r} -- run this under Python 3.10"
        )
    code = marshal.loads(data[16:])
    if not isinstance(code, types.CodeType):
        raise ParseError("no code object in .pyc")
    return code


def is_class_body(const: object) -> bool:
    """Class bodies run unoptimised; functions and comprehensions do not."""
    return (
        isinstance(const, types.CodeType)
        and not (const.co_flags & CO_OPTIMIZED)
        and not const.co_name.startswith("<")
    )


def is_function(const: object) -> bool:
    return (
        isinstance(const, types.CodeType)
        and bool(const.co_flags & CO_OPTIMIZED)
        and const.co_name not in SKIPPED_CODE_NAMES
    )


def signature(code: types.CodeType) -> str:
    """Rebuild a parameter list from a code object.

    Defaults live in the *caller's* MAKE_FUNCTION operands, not in the code
    object, so every parameter is emitted as optional.
    """
    n_pos = code.co_argcount
    n_kw = code.co_kwonlyargcount
    names = list(code.co_varnames[: n_pos + n_kw])
    positional, keyword_only = names[:n_pos], names[n_pos:]

    parts: list[str] = []
    for i, name in enumerate(positional):
        if i == 0 and name in ("self", "cls"):
            parts.append(name)
        else:
            parts.append(f"{name}: Any = ...")

    idx = n_pos + n_kw
    if code.co_flags & CO_VARARGS:
        parts.append(f"*{code.co_varnames[idx]}: Any")
        idx += 1
    elif keyword_only:
        parts.append("*")

    parts.extend(f"{name}: Any = ..." for name in keyword_only)

    if code.co_flags & CO_VARKEYWORDS:
        parts.append(f"**{code.co_varnames[idx]}: Any")

    return ", ".join(parts)


def method_decorator(code: types.CodeType) -> str | None:
    """Guess the binding from the first parameter name."""
    first = code.co_varnames[0] if code.co_argcount else None
    if first == "cls":
        return "@classmethod"
    if first != "self":
        return "@staticmethod"
    return None


# --------------------------------------------------------------------------
# structure extraction
# --------------------------------------------------------------------------


def docstring_of(code: types.CodeType) -> str | None:
    """A *function* keeps its docstring as the first constant."""
    first = code.co_consts[0] if code.co_consts else None
    return first if isinstance(first, str) else None


def string_assignments(body: types.CodeType, names: tuple[str, ...]) -> dict[str, str]:
    """Collect ``name = "some string"`` assignments from a class body.

    A class body's first constant is its ``__qualname__``, not its docstring --
    the docstring is a normal assignment to ``__doc__``, so read it the same way
    as any other string attribute.
    """
    found: dict[str, str] = {}
    last_string: str | None = None

    for instr in dis.get_instructions(body):
        if instr.opname == "LOAD_CONST":
            last_string = instr.argval if isinstance(instr.argval, str) else None
        elif instr.opname == "STORE_NAME":
            if instr.argval in names and last_string is not None:
                found.setdefault(instr.argval, last_string)
            last_string = None

    return found


def help_url_of(strings: dict[str, str]) -> str | None:
    """Build the help link an entity class records for itself.

    ``_helpFilePython`` is the Python reference page; ``_helpFile`` is the
    general one for the same entity, used only as a fallback.
    """
    for attribute in HELP_ATTRIBUTES:
        topic = strings.get(attribute)
        if topic and topic.endswith(".htm"):
            return HELP_BASE + topic
    return None


class ClassInfo:
    def __init__(self, name: str, bases: list[str]) -> None:
        self.name = name
        self.bases = bases
        self.doc: str | None = None
        self.help_url: str | None = None
        self.attributes: list[tuple[str, str]] = []
        self.methods: list[tuple[str, types.CodeType]] = []
        self.nested: list[ClassInfo] = []


def extract_classes(code: types.CodeType) -> list[ClassInfo]:
    """Find every class created in this code object, with its base names.

    3.10 emits ``LOAD_BUILD_CLASS; LOAD_CONST <body>; LOAD_CONST <name>;
    MAKE_FUNCTION; LOAD_CONST <name>; <bases...>; CALL_FUNCTION n``.
    """
    instructions = list(dis.get_instructions(code))
    bodies_by_name: dict[str, types.CodeType] = {
        const.co_name: const for const in code.co_consts if is_class_body(const)
    }

    classes: list[ClassInfo] = []
    for i, instr in enumerate(instructions):
        if instr.opname != "LOAD_BUILD_CLASS":
            continue

        body: types.CodeType | None = None
        class_name: str | None = None
        bases: list[str] = []
        pending: str | None = None

        for follow in instructions[i + 1 : i + 40]:
            if follow.opname.startswith("CALL"):
                break
            if follow.opname == "LOAD_CONST":
                if isinstance(follow.argval, types.CodeType) and body is None:
                    body = follow.argval
                elif isinstance(follow.argval, str) and class_name is None and body is not None:
                    class_name = follow.argval
            elif follow.opname in ("LOAD_NAME", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_FAST"):
                if class_name is not None:
                    if pending:
                        bases.append(pending)
                    pending = follow.argval
            elif follow.opname == "LOAD_ATTR" and pending:
                pending = f"{pending}.{follow.argval}"
        if pending:
            bases.append(pending)

        if body is None or class_name is None:
            continue
        body = bodies_by_name.get(class_name, body)
        classes.append(build_class(class_name, bases, body))

    return classes


def build_class(name: str, bases: list[str], body: types.CodeType) -> ClassInfo:
    info = ClassInfo(name, bases)
    strings = string_assignments(body, ("__doc__",) + HELP_ATTRIBUTES)
    info.doc = strings.get("__doc__")
    info.help_url = help_url_of(strings)
    info.nested = extract_classes(body)
    nested_names = {nested.name for nested in info.nested}

    for const in body.co_consts:
        if is_function(const) and const.co_name not in nested_names:
            info.methods.append((const.co_name, const))

    method_names = {method for method, _ in info.methods}
    for attr_name, attr_type in extract_attributes(body):
        if attr_name not in method_names and attr_name not in nested_names:
            info.attributes.append((attr_name, attr_type))

    return info


def extract_attributes(body: types.CodeType) -> list[tuple[str, str]]:
    """Recover class-body assignments, typed from the descriptor used.

    Entities declare properties both bare (``Double(...)``) and qualified
    (``mbd.Double(...)``), so attribute loads count too.
    """
    found: list[tuple[str, str]] = []
    last_call_name: str | None = None
    last_const: object = None
    seen: set[str] = set()

    for instr in dis.get_instructions(body):
        if instr.opname in ("LOAD_NAME", "LOAD_GLOBAL", "LOAD_ATTR", "LOAD_METHOD"):
            if instr.argval in PROPERTY_TYPES:
                last_call_name = instr.argval
        elif instr.opname == "LOAD_CONST":
            last_const = instr.argval
        elif instr.opname == "STORE_NAME":
            name = instr.argval
            if not name.startswith("_") and name not in seen:
                if last_call_name:
                    found.append((name, PROPERTY_TYPES[last_call_name]))
                elif isinstance(last_const, (str, bool, int, float)):
                    found.append((name, type(last_const).__name__))
                else:
                    found.append((name, "Any"))
                seen.add(name)
            last_call_name = None
            last_const = None

    return found


def extract_functions(code: types.CodeType, class_names: set[str]) -> list[tuple[str, types.CodeType]]:
    return [
        (const.co_name, const)
        for const in code.co_consts
        if is_function(const) and const.co_name not in class_names
    ]


def absolute_module(name: str, level: int, package: str) -> str:
    """Turn a possibly relative import into an absolute dotted module."""
    if not level:
        return name
    base = package
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    return f"{base}.{name}" if name else base


def extract_imports(
    code: types.CodeType, package: str
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Read the module's imports.

    Returns ``({local_name: (module, original_name)}, [star_imported_modules])``,
    with relative imports resolved against ``package``. The MotionView packages
    lean on ``from .Object import *``, which is what makes
    ``from hw.mview.mbd import Point`` work at runtime.
    """
    from_imports: dict[str, tuple[str, str]] = {}
    star_imports: list[str] = []

    instructions = list(dis.get_instructions(code))
    module: str | None = None
    imported: str | None = None

    for i, instr in enumerate(instructions):
        if instr.opname == "IMPORT_NAME":
            level = 0
            for previous in reversed(instructions[max(0, i - 2) : i]):
                if previous.opname == "LOAD_CONST" and isinstance(previous.argval, int):
                    level = previous.argval
                    break
            module = absolute_module(instr.argval, level, package)
            imported = None
        elif instr.opname == "IMPORT_STAR" and module:
            star_imports.append(module)
        elif instr.opname == "IMPORT_FROM":
            imported = instr.argval
        elif instr.opname == "STORE_NAME" and module:
            if imported:
                from_imports[instr.argval] = (module, imported)
            imported = None

    return from_imports, star_imports


# --------------------------------------------------------------------------
# stub emission
# --------------------------------------------------------------------------


class Module:
    def __init__(self, dotted: str, out_path: Path, code: types.CodeType, is_package: bool) -> None:
        self.dotted = dotted
        self.out_path = out_path
        self.is_package = is_package
        self.package = dotted if is_package else dotted.rpartition(".")[0]
        self.classes = extract_classes(code)
        class_names = {cls.name for cls in self.classes}
        self.functions = extract_functions(code, class_names)
        self.from_imports, self.star_imports = extract_imports(code, self.package)


def render(module: Module, index: dict[str, str]) -> str:
    """Render one stub. ``index`` maps class name -> module that defines it."""
    lines = [
        '"""Auto-generated stub -- see scripts/gen_altair_stubs.py. Do not edit."""',
        "",
        "from typing import Any",
    ]

    own_classes = {cls.name for cls in module.classes}
    needed: dict[str, tuple[str, str]] = {}

    def resolve(base: str) -> str | None:
        """Keep a base only if it lands on a class we actually generated."""
        tail = base.split(".")[-1]
        if tail in own_classes:
            return tail
        # The module's own import table is authoritative; the global index is
        # a fallback and can collide across the ~2000 classes in the tree.
        origin = module.from_imports.get(tail)
        if origin is None:
            source = index.get(tail)
            origin = (source, tail) if source else None
        if origin is None or origin[0] == module.dotted:
            return None
        needed[tail] = origin
        return tail

    rendered: list[str] = []
    for cls in module.classes:
        rendered.extend(render_class(cls, resolve, index))

    for name, code in module.functions:
        if name.startswith("_") and not name.startswith("__"):
            continue
        doc = render_docstring(docstring_of(code), "    ")
        if doc:
            rendered.append(f"def {name}({signature(code)}) -> Any:")
            rendered.extend(doc)
            rendered.append("    ...")
        else:
            rendered.append(f"def {name}({signature(code)}) -> Any: ...")
        rendered.append("")

    # Packages re-export their submodules' contents via `from .X import *`;
    # reproducing that is what makes `from hw.mview.mbd import Point` resolve.
    if module.is_package:
        lines.extend(f"from {source} import *" for source in module.star_imports)

    for local, (source, original) in sorted(needed.items()):
        if local == original:
            lines.append(f"from {source} import {local}")
        else:
            lines.append(f"from {source} import {original} as {local}")

    lines.append("")
    lines.extend(rendered)
    return "\n".join(lines).rstrip() + "\n"


def render_docstring(text: str | None, indent: str, help_url: str | None = None) -> list[str]:
    """Emit a triple-quoted docstring, escaped so it cannot break the stub."""
    parts = [part for part in (inspect.cleandoc(text) if text else "", ) if part.strip()]
    if help_url:
        parts.append(f"Altair help: {help_url}")

    content = "\n\n".join(parts).replace("\\", "\\\\").replace('"""', r"\"\"\"")
    body = content.splitlines()
    if not body:
        return []
    if len(body) == 1:
        return [f'{indent}"""{body[0]}"""']

    lines = [f'{indent}"""{body[0]}']
    lines.extend(f"{indent}{line}".rstrip() for line in body[1:])
    lines.append(f'{indent}"""')
    return lines


def render_class(cls: ClassInfo, resolve, index: dict[str, str], indent: str = "") -> list[str]:
    bases = [resolved for base in cls.bases if (resolved := resolve(base))]
    header = f"{indent}class {cls.name}({', '.join(bases)}):" if bases else f"{indent}class {cls.name}:"
    lines = [header]
    lines.extend(render_docstring(cls.doc, indent + "    ", cls.help_url))

    body: list[str] = []
    for nested in cls.nested:
        body.extend(render_class(nested, resolve, index, indent + "    "))
    for attr_name, attr_type in cls.attributes:
        body.append(f"{indent}    {attr_name}: {attr_type}")
    for method_name, code in cls.methods:
        if method_name.startswith("_") and method_name != "__init__":
            continue
        decorator = method_decorator(code)
        if decorator:
            body.append(f"{indent}    {decorator}")
        returns = "None" if method_name == "__init__" else "Any"
        header = f"{indent}    def {method_name}({signature(code)}) -> {returns}:"
        doc = render_docstring(docstring_of(code), indent + "        ")
        if doc:
            body.append(header)
            body.extend(doc)
            body.append(f"{indent}        ...")
        else:
            body.append(f"{header} ...")

    lines.extend(body or [f"{indent}    ..."])
    lines.append("")
    return lines


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def collect(package_root: Path, package: str) -> tuple[list[Module], list[tuple[Path, str]]]:
    modules: list[Module] = []
    failures: list[tuple[Path, str]] = []

    for pyc in sorted(package_root.rglob("*.pyc")):
        relative = pyc.relative_to(package_root.parent)
        parts = list(relative.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
            out_path = Path(*relative.parts[:-1], "__init__.pyi")
        else:
            out_path = relative.with_suffix(".pyi")
        dotted = ".".join(parts) or package

        try:
            modules.append(Module(dotted, out_path, load_code(pyc), is_package))
        except ParseError as exc:
            failures.append((pyc, str(exc)))
            if "magic" in str(exc):  # wrong interpreter: fail fast, not 500 times
                raise
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures.append((pyc, f"{type(exc).__name__}: {exc}"))

    return modules, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--altair-root", type=Path, default=DEFAULT_ALTAIR_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "typings",
        help="stub output directory (Pylance defaults to ./typings)",
    )
    parser.add_argument("--clean", action="store_true", help="wipe the package tree in --out first")
    args = parser.parse_args()

    package_root = args.altair_root / PACKAGE_PARENT / PACKAGE
    if not package_root.is_dir():
        print(f"error: {package_root} not found -- pass --altair-root", file=sys.stderr)
        return 1

    modules, failures = collect(package_root, PACKAGE)
    if not modules:
        print("error: no modules parsed", file=sys.stderr)
        return 1

    # class name -> defining module, so bases can be imported across stubs
    index: dict[str, str] = {}
    for module in modules:
        for cls in module.classes:
            index.setdefault(cls.name, module.dotted)

    target = args.out / PACKAGE
    if args.clean and target.exists():
        shutil.rmtree(target)

    written = 0
    invalid: list[tuple[Path, str]] = []
    for module in modules:
        text = render(module, index)
        try:
            ast.parse(text)
        except SyntaxError as exc:
            invalid.append((module.out_path, f"line {exc.lineno}: {exc.msg}"))
            continue
        destination = args.out / module.out_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        written += 1

    print(f"wrote {written} stub(s) to {args.out}")
    print(f"indexed {len(index)} classes")
    for path, reason in failures:
        print(f"  skipped {path.name}: {reason}")
    for path, reason in invalid:
        print(f"  invalid syntax, not written: {path} ({reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
