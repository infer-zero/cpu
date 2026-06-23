# infer-cpu

CPU inference backend in Zig, with multithreaded operators and P-core thread pool.

This was build with AI assistance specially for optinization and quantization support.

## Operators

Per datatype kernel operations.

- `ops.bf16`: BF16 matmul, RMSNorm, RoPE, softmax, weighted sum, residual add.
- `ops.q8_0`: Q8_0 quantized matmul + dequant helpers (32-element blocks, f16 scale).
- `ops.q4_0`: Q4_0 quantized matmul + dequant helpers (32-element blocks, f16 scale, 4-bit weights).
- `ops.common`: Shared utilities (softmax, weighted sum, `scaledAdd`).

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

A typical CPU variant uses the thread pool plus one of the data_type op families:

```zig
const cpu = @import("cpu");

// Pin a thread pool to the P-cores (falls back to logical CPU count).
const n_threads = cpu.parallel.threadCount(io);
var group: std.Io.Group = .init;
const exec = Executor{ .io = io, .group = &group, .n_threads = n_threads };

// Run matmul on Q8_0 using R4 packing on the executor
cpu.ops.q8.matmulQ8R4(exec, q_vals, q_scales, attn_query_projection, batch_q, dim, query_dim, position);

```

## License

MIT
