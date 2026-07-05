"""Optional native backend for the rank-5 residue kernel."""

from __future__ import annotations

import ctypes
import json
import platform
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

from .delta_mod import DerivOrders
from .sparse_mod import SparsePoly


class NativeResidueUnavailable(RuntimeError):
    """Raised when the optional native residue library is not built."""


def native_library_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        name = "libr5g2higgs_residue_kernel.dylib"
    elif system == "Windows":
        name = "r5g2higgs_residue_kernel.dll"
    else:
        name = "libr5g2higgs_residue_kernel.so"
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "native" / "residue_kernel" / "target" / "release" / name


@lru_cache(maxsize=1)
def _load_native_library() -> ctypes.CDLL:
    path = native_library_path()
    if not path.exists():
        raise NativeResidueUnavailable(
            f"native residue library is not built: {path}. "
            "Run `cargo build --release --manifest-path native/residue_kernel/Cargo.toml`."
        )
    lib = ctypes.CDLL(str(path))
    lib.rust_residue_poly_mod_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint64,
    ]
    lib.rust_residue_poly_mod_batch.restype = ctypes.c_uint64
    lib.rust_residue_product_mod_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint64,
    ]
    lib.rust_residue_product_mod_batch.restype = ctypes.c_uint64
    lib.rust_residue_products_sum_mod_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint64,
    ]
    lib.rust_residue_products_sum_mod_batch.restype = ctypes.c_uint64
    lib.rust_residue_products_sum_profile_json.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint64,
    ]
    lib.rust_residue_products_sum_profile_json.restype = ctypes.c_void_p
    lib.rust_residue_free_string.argtypes = [ctypes.c_void_p]
    lib.rust_residue_free_string.restype = None
    lib.rust_residue_clear_caches.argtypes = []
    lib.rust_residue_clear_caches.restype = None
    return lib


def native_available() -> bool:
    try:
        _load_native_library()
    except (NativeResidueUnavailable, OSError, AttributeError):
        return False
    return True


def clear_native_caches() -> None:
    _load_native_library().rust_residue_clear_caches()


def _pack_poly(poly: SparsePoly, p: int):
    items = [(alpha, int(coeff) % p) for alpha, coeff in poly.items() if int(coeff) % p]
    length = len(items)
    exp_array = (ctypes.c_int32 * (4 * length))()
    coeff_array = (ctypes.c_uint64 * length)()
    for idx, (alpha, coeff) in enumerate(items):
        exp_array[4 * idx] = int(alpha[0])
        exp_array[4 * idx + 1] = int(alpha[1])
        exp_array[4 * idx + 2] = int(alpha[2])
        exp_array[4 * idx + 3] = int(alpha[3])
        coeff_array[idx] = int(coeff)
    return exp_array, coeff_array, length


def _pack_deriv_orders(deriv_orders: DerivOrders):
    return (ctypes.c_int32 * 4)(
        int(deriv_orders[0]),
        int(deriv_orders[1]),
        int(deriv_orders[2]),
        int(deriv_orders[3]),
    )


def residue_poly_batch_native(poly: SparsePoly, deriv_orders: DerivOrders, p: int) -> int:
    exp_array, coeff_array, length = _pack_poly(poly, p)
    if not length:
        return 0

    deriv_array = _pack_deriv_orders(deriv_orders)
    return int(
        _load_native_library().rust_residue_poly_mod_batch(
            exp_array,
            coeff_array,
            length,
            deriv_array,
            int(p),
        )
    )


def residue_product_native(
    left: SparsePoly,
    right: SparsePoly,
    deriv_orders: DerivOrders,
    p: int,
) -> int:
    left_exp, left_coeff, left_len = _pack_poly(left, p)
    right_exp, right_coeff, right_len = _pack_poly(right, p)
    if not left_len or not right_len:
        return 0

    deriv_array = _pack_deriv_orders(deriv_orders)
    return int(
        _load_native_library().rust_residue_product_mod_batch(
            left_exp,
            left_coeff,
            left_len,
            right_exp,
            right_coeff,
            right_len,
            deriv_array,
            int(p),
        )
    )


def _pack_products_sum_inputs(
    left: SparsePoly,
    tasks: Sequence[Tuple[DerivOrders, SparsePoly]],
    p: int,
) :
    if not tasks:
        return None

    left_exp, left_coeff, left_len = _pack_poly(left, p)
    if not left_len:
        return None

    normalized_tasks = []
    total_right_len = 0
    for deriv_orders, right in tasks:
        items = [(alpha, int(coeff) % p) for alpha, coeff in right.items() if int(coeff) % p]
        normalized_tasks.append((deriv_orders, items))
        total_right_len += len(items)
    if not total_right_len:
        return None

    task_count = len(normalized_tasks)
    right_exp = (ctypes.c_int32 * (4 * total_right_len))()
    right_coeff = (ctypes.c_uint64 * total_right_len)()
    right_offsets = (ctypes.c_size_t * (task_count + 1))()
    deriv_array = (ctypes.c_int32 * (4 * task_count))()

    cursor = 0
    right_offsets[0] = 0
    for task_idx, (deriv_orders, items) in enumerate(normalized_tasks):
        deriv_array[4 * task_idx] = int(deriv_orders[0])
        deriv_array[4 * task_idx + 1] = int(deriv_orders[1])
        deriv_array[4 * task_idx + 2] = int(deriv_orders[2])
        deriv_array[4 * task_idx + 3] = int(deriv_orders[3])
        for alpha, coeff in items:
            right_exp[4 * cursor] = int(alpha[0])
            right_exp[4 * cursor + 1] = int(alpha[1])
            right_exp[4 * cursor + 2] = int(alpha[2])
            right_exp[4 * cursor + 3] = int(alpha[3])
            right_coeff[cursor] = int(coeff)
            cursor += 1
        right_offsets[task_idx + 1] = cursor

    return (
        left_exp,
        left_coeff,
        left_len,
        right_exp,
        right_coeff,
        right_offsets,
        task_count,
        deriv_array,
    )


def residue_products_sum_native(
    left: SparsePoly,
    tasks: Sequence[Tuple[DerivOrders, SparsePoly]],
    p: int,
) -> int:
    packed = _pack_products_sum_inputs(left, tasks, p)
    if packed is None:
        return 0
    (
        left_exp,
        left_coeff,
        left_len,
        right_exp,
        right_coeff,
        right_offsets,
        task_count,
        deriv_array,
    ) = packed

    return int(
        _load_native_library().rust_residue_products_sum_mod_batch(
            left_exp,
            left_coeff,
            left_len,
            right_exp,
            right_coeff,
            right_offsets,
            task_count,
            deriv_array,
            int(p),
        )
    )


def residue_products_sum_profile_native(
    left: SparsePoly,
    tasks: Sequence[Tuple[DerivOrders, SparsePoly]],
    p: int,
) -> dict:
    packed = _pack_products_sum_inputs(left, tasks, p)
    if packed is None:
        return {"value": 0}
    (
        left_exp,
        left_coeff,
        left_len,
        right_exp,
        right_coeff,
        right_offsets,
        task_count,
        deriv_array,
    ) = packed

    lib = _load_native_library()
    ptr = lib.rust_residue_products_sum_profile_json(
        left_exp,
        left_coeff,
        left_len,
        right_exp,
        right_coeff,
        right_offsets,
        task_count,
        deriv_array,
        int(p),
    )
    if not ptr:
        raise RuntimeError("native residue profiler returned a null JSON pointer")
    try:
        return json.loads(ctypes.string_at(ptr).decode("utf-8"))
    finally:
        lib.rust_residue_free_string(ptr)
