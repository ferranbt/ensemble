"""Declaring a tool.

One decorator does everything a tool needs:

    app = app_for("rmsd")
    image = with_local_sources(base_image())

    @tool("rmsd/compare_structures", image=image, timeout=15 * 60)
    def compare_structures(rt, reference_uri: str, ...) -> dict:
        rt.storage.download(reference_uri, ...)
        return {"run_id": rt.run_id, ...}

It hands the function a `Runtime`, records the call, and makes it a Modal
function. The Modal app comes from the address, so the app name and the name a
workflow uses cannot drift apart — they are the same string, written once.

The module still binds `app` at the top because `modal deploy` imports a file
and looks for a module-level app. `app_for` returns the same object the
decorator uses, so that line is a handle, not a second declaration.

This is the only file besides `images.py` that a tool needs Modal for, and
tool bodies never import it: a body takes `rt` and returns a dict.
"""

from __future__ import annotations

import functools
import inspect
import json
from typing import Any, Callable

import modal

from tools.common.images import SECRETS
from tools.common.records import Runtime, records

# Modal's own defaults are generous; 15 minutes suits the CPU tools, and
# anything longer says so explicitly.
DEFAULT_TIMEOUT = 15 * 60

# What Modal's CLI can turn a command-line word into. A parameter of any other
# type is declared to Modal as a string and decoded as JSON on the way in,
# which is what a hand-written entrypoint used to do for `chains` and
# `variants`. Without it `modal run` refuses the whole function.
SCALARS = {"str", "int", "float", "bool"}

_APPS: dict[str, modal.App] = {}
_VOLUMES: dict[str, modal.Volume] = {}


def app_for(name: str) -> modal.App:
    """The Modal app of that name, made once per process.

    Shared so that several tools in one app, as ProteinMPNN and ESM have, all
    attach to the same object, and so the module-level `app` a deploy looks for
    is the one the decorator used.
    """
    if name not in _APPS:
        _APPS[name] = modal.App(name)
    return _APPS[name]


def _volume(name: str) -> modal.Volume:
    """A named persistent volume, made once per process."""
    if name not in _VOLUMES:
        _VOLUMES[name] = modal.Volume.from_name(name, create_if_missing=True)
    return _VOLUMES[name]


def _annotation(parameter: inspect.Parameter) -> str:
    """A parameter's annotation as written, or "" when it has none.

    Text rather than a type, because these modules use
    `from __future__ import annotations` and never evaluate them.
    """
    if parameter.annotation is inspect.Parameter.empty:
        return ""
    return str(parameter.annotation)


def _as_text(parameter: inspect.Parameter) -> inspect.Parameter:
    """The same parameter, declared as a string."""
    return parameter.replace(annotation=str)


def _decoded(kwargs: dict[str, Any], structured: set[str]) -> dict[str, Any]:
    """Read JSON out of the structured arguments that arrived as text.

    A caller in Python passes the real object and it is left alone; only a
    string is decoded, and only for a parameter whose own type is not a
    string. Text that is not JSON is passed through untouched, so a mistyped
    argument fails inside the tool, against the tool's own validation, rather
    than here.
    """
    decoded = dict(kwargs)
    for name in structured & set(decoded):
        value = decoded[name]
        if not isinstance(value, str):
            continue
        try:
            decoded[name] = json.loads(value)
        except json.JSONDecodeError:
            pass
    return decoded


def tool(
    address: str,
    *,
    image: modal.Image,
    gpu: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    cache: str = "",
    cache_path: str = "",
    secrets: list[modal.Secret] | None = None,
    record: bool = True,
) -> Callable:
    """Declare a function as a tool and a Modal function.

    Args:
        address: `"<app>/<function>"`, exactly as a workflow names it. The app
            half names the Modal app and the other half must match the
            function's own name, so renaming one without the other fails here
            rather than leaving a workflow pointing at nothing.
        image: What the tool runs inside.
        gpu: Which accelerator, or None for CPU.
        timeout: Seconds before Modal kills the call.
        cache: Name of a persistent volume for model weights, so only the
            first call pays to download them. Needs `cache_path`.
        cache_path: Where that volume is mounted.
        secrets: Defaults to the three every tool uses.
        record: Write a database row per call.

    Returns:
        A Modal function. Its visible signature excludes the runtime, so
        callers, Modal and every workflow pass only the tool's own arguments.

    Raises:
        ValueError: if the address is malformed, disagrees with the function's
            name, takes no arguments, or names a cache without a path.
    """
    if address.count("/") != 1 or not all(address.split("/")):
        raise ValueError(
            f"Tool address {address!r} must be '<app>/<function>', matching "
            f"how a workflow names it."
        )
    app_name, fn_name = address.split("/")
    if bool(cache) != bool(cache_path):
        raise ValueError(
            f"{address} gives cache={cache!r} and cache_path={cache_path!r}; a "
            f"volume needs both a name and somewhere to be mounted."
        )

    def decorate(fn: Callable) -> Callable:
        if fn.__name__ != fn_name:
            raise ValueError(
                f"{address} names the function {fn_name!r} but it is called "
                f"{fn.__name__!r}. A workflow addresses it by that name, so "
                f"the two cannot differ."
            )

        signature = inspect.signature(fn)
        parameters = list(signature.parameters.values())
        if not parameters:
            raise ValueError(
                f"{address} takes no arguments. A tool's first parameter is "
                f"the runtime it is handed."
            )

        volume = _volume(cache) if cache else None
        structured = {
            p.name for p in parameters[1:] if _annotation(p) not in SCALARS
        }

        @functools.wraps(fn)
        def supplied(*args: Any, **kwargs: Any) -> Any:
            runtime = Runtime.create(fn_name, cache=volume)
            result = fn(runtime, *args, **_decoded(kwargs, structured))
            if isinstance(result, dict):
                result = {"run_id": runtime.run_id, **result}
            # Printed so that `modal run <file>::<function>` shows the answer.
            # Modal discards a function's return value, so without this the
            # only way to see a result from the command line was a
            # hand-written entrypoint per tool. It lands in that call's own log
            # stream, so a fan-out keeps each result beside the work that
            # produced it.
            print(json.dumps(result, indent=2, default=repr), flush=True)
            return result

        # Without this, `inspect.signature` follows `__wrapped__` back to `fn`
        # and reports a runtime parameter no caller supplies. Modal reads the
        # signature, so it would be wrong about the tool's arguments.
        supplied.__signature__ = signature.replace(
            parameters=[_as_text(p) if p.name in structured else p
                        for p in parameters[1:]]
        )
        # Modal's CLI takes parameter names from the signature but their types
        # from `__annotations__`, which `functools.wraps` copied from the tool
        # untouched. Both have to agree or the runtime is told one thing and
        # the command line another.
        supplied.__annotations__ = {
            name: str if name in structured else hint
            for name, hint in supplied.__annotations__.items()
            if name != parameters[0].name
        }

        wrapped = records(address)(supplied) if record else supplied

        options: dict[str, Any] = {
            "image": image,
            "timeout": timeout,
            "secrets": SECRETS if secrets is None else secrets,
        }
        if gpu:
            options["gpu"] = gpu
        if volume is not None:
            options["volumes"] = {cache_path: volume}

        return app_for(app_name).function(**options)(wrapped)

    return decorate
