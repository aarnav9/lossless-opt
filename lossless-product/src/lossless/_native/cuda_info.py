"""Isolated CUDA Driver API capability query. No context, allocations or launches.

Attribute numbers are the public CUdevice_attribute ABI from cuda.h. Attributes
unsupported by a driver remain null with their numeric CUDA result code.
"""

import ctypes as C
import json

ATTRIBUTES = {
    "max_threads_per_block": 1,
    "max_block_dim_x": 2,
    "max_block_dim_y": 3,
    "max_block_dim_z": 4,
    "max_grid_dim_x": 5,
    "max_grid_dim_y": 6,
    "max_grid_dim_z": 7,
    "shared_memory_per_block_bytes": 8,
    "constant_memory_bytes": 9,
    "warp_size": 10,
    "registers_per_block_32bit": 12,
    "clock_rate_khz_reported": 13,
    "multiprocessor_count": 16,
    "integrated": 18,
    "concurrent_kernels": 31,
    "memory_clock_rate_khz_reported": 36,
    "global_memory_bus_width_bits": 37,
    "l2_cache_bytes": 38,
    "max_threads_per_multiprocessor": 39,
    "async_engine_count": 40,
    "unified_addressing": 41,
    "compute_capability_major": 75,
    "compute_capability_minor": 76,
    "shared_memory_per_multiprocessor_bytes": 81,
    "registers_per_multiprocessor_32bit": 82,
    "managed_memory": 83,
    "shared_memory_per_block_optin_bytes": 97,
    "max_blocks_per_multiprocessor": 106,
}


def discover():
    try:
        driver = C.CDLL("libcuda.so.1")
    except OSError:
        return {"status": "unavailable", "reason": "cuda_driver_library_not_found"}
    try:
        driver.cuInit.argtypes = [C.c_uint]
        driver.cuDeviceGetCount.argtypes = [C.POINTER(C.c_int)]
        driver.cuDriverGetVersion.argtypes = [C.POINTER(C.c_int)]
        driver.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
        driver.cuDeviceGetName.argtypes = [C.c_char_p, C.c_int, C.c_int]
        driver.cuDeviceGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
        driver.cuDeviceTotalMem_v2.argtypes = [C.POINTER(C.c_size_t), C.c_int]
        code = driver.cuInit(0)
        if code:
            return {"status": "unavailable", "reason": "cuInit_failed", "cuda_error_code": code}
        count, version = C.c_int(), C.c_int()
        code = driver.cuDeviceGetCount(C.byref(count))
        if code:
            return {
                "status": "unavailable",
                "reason": "device_count_failed",
                "cuda_error_code": code,
            }
        version_code = driver.cuDriverGetVersion(C.byref(version))
        devices = []
        for index in range(count.value):
            device = C.c_int()
            code = driver.cuDeviceGet(C.byref(device), index)
            if code:
                devices.append({"index": index, "status": "unavailable", "cuda_error_code": code})
                continue
            name, total = C.create_string_buffer(256), C.c_size_t()
            name_code = driver.cuDeviceGetName(name, 256, device)
            mem_code = driver.cuDeviceTotalMem_v2(C.byref(total), device)
            values, errors = {}, {}
            for key, attribute in ATTRIBUTES.items():
                value = C.c_int()
                code = driver.cuDeviceGetAttribute(C.byref(value), attribute, device)
                values[key] = value.value if code == 0 else None
                if code:
                    errors[key] = code
            devices.append(
                {
                    "index": index,
                    "status": "ok",
                    "name": name.value.decode(errors="replace") if name_code == 0 else None,
                    "total_memory_bytes": total.value if mem_code == 0 else None,
                    "attributes": values,
                    "attribute_error_codes": errors,
                }
            )
        return {
            "status": "ok" if devices else "unavailable",
            "devices": devices,
            "driver_api_version": version.value if version_code == 0 else None,
            "source": "CUDA Driver API cuDeviceGetAttribute / cuDeviceTotalMem_v2",
            "scope": "Capacity/limits only; opt-in shared memory requires kernel configuration. "
            "No measured bandwidth, free memory, register use or occupancy. Linux CUDA path not yet hardware-validated in this project.",
        }
    except AttributeError:
        return {"status": "unavailable", "reason": "missing_driver_api_symbol"}


if __name__ == "__main__":
    print(json.dumps(discover(), indent=2))
