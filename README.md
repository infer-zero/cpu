# cpu

CPU inference backend with multithreaded operators and physical core detection.

## Modules

- **thread_pool** — Thread pool initialization with Linux P-core detection. Filters by CPU frequency to pin work to performance cores, with automatic fallback to logical CPU count.
- **ops** — Operator implementations for multiple data types:
  - `ops.bf16` — BF16 operations
  - `ops.q8_0` — Q8_0 quantized operations
  - `ops.q4_0` — Q4_0 quantized operations
  - `ops.common` — Shared utilities
- **mem** — Memory checking utilities. Estimates model size (supports both GGUF files and HuggingFace directories) and checks available system memory via `/proc/meminfo`.

## Usage

```bash
zig fetch --save git+https://github.com/infer-zero/cpu
```

Then in your `build.zig`:

```zig
const cpu_dep = b.dependency("infer_cpu", .{ .target = target, .optimize = optimize });
my_mod.addImport("cpu", cpu_dep.module("infer_cpu"));
```

```zig
const cpu = @import("cpu");

// Create a thread pool pinned to P-cores
const pool = cpu.thread_pool.initPool(allocator);
defer if (pool) |p| p.deinit();

// Check if model fits in memory
const info = try cpu.mem.checkModelFitsInMemory("/path/to/model");
```

## Dependencies

- [base](https://github.com/infer-zero/base) — Core inference abstractions

## License

MIT
