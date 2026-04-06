#!/usr/bin/env python3
"""
vllm_build_all_modelinfo_caches.py

Pre-generate vLLM model-info caches by iterating over the vLLM model registry.

Usage:
  export VLLM_CACHE_ROOT=$PWD/.vllm_cache
  python vllm_build_all_modelinfo_caches.py

Optional:
  python vllm_build_all_modelinfo_caches.py --verbose
  python vllm_build_all_modelinfo_caches.py --arch Llama4ForConditionalGeneration
  python vllm_build_all_modelinfo_caches.py --no-load-plugins
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Iterable

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def color(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def okay(text: str = "[OK]") -> str:
    return color(text, "32")   # green

def fail(text: str = "[FAIL]") -> str:
    return color(text, "31")   # red

def skip(text: str = "[SKIP]") -> str:
    return color(text, "33")   # yellow

def info(text: str = "[INFO]") -> str:
    return color(text, "36")   # cyan


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print cache paths and full skip/fail reasons.",
    )
    ap.add_argument(
        "--arch",
        action="append",
        default=[],
        metavar="ARCH",
        help=(
            "Only build caches for matching architecture names. "
            "May be passed multiple times."
        ),
    )
    ap.add_argument(
        "--no-load-plugins",
        action="store_true",
        help="Do not call vLLM general plugin loading before reading the registry.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List discovered registry entries and exit without building caches.",
    )
    return ap.parse_args()


def architecture_selected(architecture: str, filters: list[str]) -> bool:
    if not filters:
        return True
    architecture_lower = architecture.lower()
    return any(token.lower() in architecture_lower for token in filters)


def import_registry_module():
    try:
        import vllm.model_executor.models.registry as r
    except Exception as e:
        print(f"ERROR: failed to import vLLM registry internals: {e!r}", file=sys.stderr)
        raise SystemExit(1) from e
    return r


def maybe_load_plugins(verbose: bool) -> None:
    try:
        from vllm.plugins import load_general_plugins
    except Exception as e:
        if verbose:
            print(f"{skip()} plugin loading unavailable ({e.__class__.__name__}: {e})")
        return

    try:
        load_general_plugins()
    except Exception as e:
        if verbose:
            print(f"{skip()} plugin loading failed ({e.__class__.__name__}: {e})")


def iter_registry_entries(r, verbose: bool) -> Iterable[tuple[str, object]]:
    """
    Yield `(architecture, registered_model)` pairs using the installed vLLM registry.

    Preferred path:
      registry.ModelRegistry.models

    Compatibility fallback for older builds:
      registry._VLLM_MODELS -> synthesize _LazyRegisteredModel entries
    """
    model_registry = getattr(r, "ModelRegistry", None)
    models = getattr(model_registry, "models", None)
    if isinstance(models, dict) and models:
        for architecture, registered_model in sorted(models.items()):
            yield architecture, registered_model
        return

    mapping = getattr(r, "_VLLM_MODELS", None)
    Lazy = getattr(r, "_LazyRegisteredModel", None)
    if not isinstance(mapping, dict) or Lazy is None:
        print(
            "ERROR: could not locate vLLM registry entries via "
            "ModelRegistry.models or _VLLM_MODELS",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if verbose:
        print("{info()} falling back to registry._VLLM_MODELS compatibility path")

    for architecture, value in sorted(mapping.items()):
        if not isinstance(value, tuple) or len(value) != 2:
            continue
        mod_relname, cls_name = value
        if not isinstance(mod_relname, str) or not isinstance(cls_name, str):
            continue
        yield architecture, Lazy(
            module_name=f"vllm.model_executor.models.{mod_relname}",
            class_name=cls_name,
        )


def get_module_hash(r, module_name: str) -> tuple[str, Path]:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"module {module_name!r} has no __file__")

    module_path = Path(module_file)
    module_bytes = module_path.read_bytes()
    module_hash = r.safe_hash(module_bytes, usedforsecurity=False).hexdigest()
    return module_hash, module_path


def is_lazy_registered_model(registered_model: object) -> bool:
    return hasattr(registered_model, "module_name") and hasattr(registered_model, "class_name")


def is_registered_model(registered_model: object) -> bool:
    return hasattr(registered_model, "model_cls") and hasattr(registered_model, "interfaces")


def build_cache_for_lazy_entry(r, architecture: str, registered_model: object) -> tuple[Path, str]:
    """
    Build and persist the cache for a lazy registry entry using vLLM's own cache file name + serializer.
    """
    ModelInfo = getattr(r, "_ModelInfo", None)
    if ModelInfo is None:
        raise RuntimeError("registry._ModelInfo is unavailable")

    module_name = registered_model.module_name
    class_name = registered_model.class_name

    module_hash, module_path = get_module_hash(r, module_name)
    model_cls = registered_model.load_model_cls()
    model_info = ModelInfo.from_model_cls(model_cls)
    registered_model._save_modelinfo_to_cache(model_info, module_hash)
    cache_path = registered_model._get_cache_dir() / registered_model._get_cache_filename()
    return cache_path, str(module_path)


def describe_registered_entry(registered_model: object) -> str:
    if is_lazy_registered_model(registered_model):
        return f"{registered_model.module_name}:{registered_model.class_name}"
    if is_registered_model(registered_model):
        model_cls = getattr(registered_model, "model_cls", None)
        if model_cls is not None:
            return f"{model_cls.__module__}:{model_cls.__name__}"
    return type(registered_model).__name__


def main() -> int:
    args = parse_args()

    cache_root = os.environ.get("VLLM_CACHE_ROOT")
    if not cache_root:
        print("ERROR: VLLM_CACHE_ROOT must be set", file=sys.stderr)
        return 3

    r = import_registry_module()

    if not args.no_load_plugins:
        maybe_load_plugins(verbose=args.verbose)

    entries = [
        (architecture, registered_model)
        for architecture, registered_model in iter_registry_entries(r, verbose=args.verbose)
        if architecture_selected(architecture, args.arch)
    ]

    if args.list:
        for architecture, registered_model in entries:
            print(f"{architecture}\t{describe_registered_entry(registered_model)}")
        return 0

    print(f"Using VLLM_CACHE_ROOT={cache_root}")
    print(f"Discovered {len(entries)} registry entries")

    success = 0
    skipped = 0
    failed = 0

    # Avoid rebuilding the same cache repeatedly when multiple architecture
    # aliases point at the same underlying `<module, class>` pair.
    seen_lazy_targets: dict[tuple[str, str], list[str]] = {}

    for architecture, registered_model in entries:
        entry_desc = describe_registered_entry(registered_model)

        if is_lazy_registered_model(registered_model):
            key = (registered_model.module_name, registered_model.class_name)
            if key in seen_lazy_targets:
                skipped += 1
                if args.verbose:
                    aliases = ", ".join(seen_lazy_targets[key])
                    print(
                        f"{skip()} {architecture} -> {entry_desc} "
                        f"(duplicate target; already built for: {aliases})"
                    )
                seen_lazy_targets[key].append(architecture)
                continue
            seen_lazy_targets[key] = [architecture]

            try:
                cache_path, module_path = build_cache_for_lazy_entry(
                    r, architecture, registered_model
                )
                success += 1
                print(f"{okay()} {architecture} -> {entry_desc}")
                if args.verbose:
                    print(f"     module: {module_path}")
                    print(f"     cache : {cache_path}")
            except Exception as e:
                failed += 1
                print(f"{fail()} {architecture} -> {entry_desc}")
                if args.verbose:
                    print(f"       {e.__class__.__name__}: {e}")
                continue
            continue

        # Pre-imported `_RegisteredModel` entries do not need lazy cache files;
        # those entries already carry inspected interface info in memory.
        if is_registered_model(registered_model):
            skipped += 1
            if args.verbose:
                print(
                    f"{skip()} {architecture} -> {entry_desc} "
                    "(already registered eagerly; no lazy cache file to write)"
                )
            continue

        skipped += 1
        if args.verbose:
            details = []
            if is_dataclass(registered_model):
                details.append("dataclass")
            print(
                f"{skip()} {architecture} -> {entry_desc} "
                f"(unsupported registry entry shape: {type(registered_model).__name__})"
            )

    print("\nSummary:")
    print(f"  success: {success}")
    print(f"  failed : {failed}")
    print(f"  skipped: {skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

