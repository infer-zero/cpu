# infer-cpu

CPU inference backend in Zig: multithreaded operators, P-core thread pool,
memory fit-checks.

This is the shared CPU backend used by every CPU model variant in the
`infer` workspace. It does not depend on `infer_runtime` — the runtime
borrows the kernels here, not the other way around.

## Parts

### Operators

These are the per-dtype kernel families every CPU variant calls into.

- `ops.bf16`: BF16 matmul, RMSNorm, RoPE, softmax, weighted sum, residual
  add. Pure scalar paths plus `@Vector` fast paths where LLVM does not
  auto-vectorize.
- `ops.q8_0`: Q8_0 quantized matmul + dequant helpers (32-element blocks,
  f16 scale).
- `ops.q4_0`: Q4_0 quantized matmul + dequant helpers (32-element blocks,
  f16 scale, 4-bit weights).
- `ops.common`: Shared utilities (softmax, weighted sum, `scaledAdd`)
  reused across the dtype families.

### Thread pool

- `thread_pool.initPool`: builds a `std.Thread.Pool` sized to the number of
  Linux performance cores. Reads `/sys/devices/system/cpu/.../cpufreq` to
  pick out P-cores by max frequency, with automatic fallback to logical
  CPU count on platforms where the data is missing.

### Memory fit-check

- `mem.checkModelFitsInMemory`: estimates the on-disk size of a model
  (GGUF file or HuggingFace directory) and compares against
  `/proc/meminfo` so a runner can fail fast before starting a load that
  would OOM.

## Usage

Fetch the library:

```bash
zig fetch --save git+https://github.com/infer-zero/cpu
```

Add the dependency in your `build.zig`:

```zig
const cpu_dep = b.dependency("infer_cpu", .{ .target = target, .optimize = optimize });
my_mod.addImport("cpu", cpu_dep.module("infer_cpu"));
```

A typical CPU variant uses the thread pool plus one of the dtype op
families:

```zig
const cpu = @import("cpu");

// Pin a thread pool to the P-cores (falls back to logical CPU count).
const pool = cpu.thread_pool.initPool(allocator);
defer if (pool) |p| p.deinit();

// Optional: bail before load if the file does not fit in RAM.
const fit = try cpu.mem.checkModelFitsInMemory("/path/to/model");
if (!fit.fits) return error.ModelTooLargeForMemory;

// BF16 matmul — pool is optional; nil pool runs single-threaded.
cpu.ops.bf16.matmul(output, weights, activations, .{
    .rows = rows,
    .cols = cols,
    .pool = pool,
});
```

## AI Usage

- The first full version of this library was hand written.
- Some kernels, fixes and Zig version migrations were AI assisted.
- Comments and docs were AI written and human edited.
- All was human reviewed.
- The design and per-dtype kernel split is my own.

## License

MIT
