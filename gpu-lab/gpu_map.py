"""Run a prewritten Map kernel on NVIDIA hardware using the Windows driver."""
import ctypes as c
from contextlib import ExitStack
from pathlib import Path


def main():
    driver = c.WinDLL("nvcuda.dll")

    def call(name, *args):
        status = getattr(driver, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    device = c.c_int()
    context = c.c_void_p()
    module = c.c_void_p()
    kernel = c.c_void_p()
    source = (c.c_int32 * 5)(-3, 0, 4, 7, 10)
    result = type(source)()
    size = c.c_size_t(c.sizeof(source))
    count = c.c_int32(len(source))
    ptx = Path(__file__).with_name("map.ptx").read_bytes()

    call("cuInit", 0)
    call("cuDeviceGet", c.byref(device), 0)
    with ExitStack() as cleanup:
        call("cuDevicePrimaryCtxRetain", c.byref(context), device)
        cleanup.callback(call, "cuDevicePrimaryCtxRelease_v2", device)
        call("cuCtxSetCurrent", context)
        cleanup.callback(call, "cuCtxSetCurrent", None)
        error_log = c.create_string_buffer(4096)
        options = (c.c_int * 2)(5, 6)
        values = (c.c_void_p * 2)(c.addressof(error_log), len(error_log))
        try:
            call("cuModuleLoadDataEx", c.byref(module), c.c_char_p(ptx), 2, options, values)
        except RuntimeError as error:
            raise RuntimeError(f"{error}: {error_log.value.decode(errors='replace')}") from error
        cleanup.callback(call, "cuModuleUnload", module)
        call("cuModuleGetFunction", c.byref(kernel), module, c.c_char_p(b"map_double"))

        src, dst = c.c_uint64(), c.c_uint64()
        for address in (src, dst):
            call("cuMemAlloc_v2", c.byref(address), size)
            cleanup.callback(call, "cuMemFree_v2", address)

        call("cuMemcpyHtoD_v2", src, source, size)
        args = (c.c_void_p * 3)(c.addressof(src), c.addressof(dst), c.addressof(count))
        # Grid: one block. Block: eight threads. Five have work.
        call("cuLaunchKernel", kernel, 1, 1, 1, 8, 1, 1, 0, None, args, None)
        call("cuCtxSynchronize")
        call("cuMemcpyDtoH_v2", result, dst, size)

    expected = [2 * x for x in source]
    if list(result) != expected:
        raise RuntimeError(f"GPU returned {list(result)}, expected {expected}")
    print("GPU Map:", list(source), "->", list(result), "| CPU reference matched")


if __name__ == "__main__":
    main()
