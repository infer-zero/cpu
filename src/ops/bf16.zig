const std = @import("std");
const common = @import("common.zig");

pub const DataType = enum { FP32, FP16, BF16 };

const Pool = std.Thread.Pool;
const WaitGroup = std.Thread.WaitGroup;

// ---- BF16-specific element-wise operations ----

pub fn f32ToBF16(src: []const f32, dst: []u16) void {
    for (src, dst) |src_val, *dst_val| {
        dst_val.* = @truncate(@as(u32, @bitCast(src_val)) >> 16);
    }
}

pub fn dotF32BF16(a: []const f32, b: []const u16) f32 {
    // f64 accumulator — used for attention QK scoring against bf16 KV cache.
    // See common.dot.
    var sum: f64 = 0.0;
    for (a, b) |a_val, b_val| {
        const bits: u32 = @as(u32, b_val) << 16;
        const b_f32: f32 = @bitCast(bits);
        sum += @as(f64, a_val) * @as(f64, b_f32);
    }
    return @floatCast(sum);
}

pub fn scaledAddBF16(output: []f32, values: []const u16, scale: f32) void {
    @setFloatMode(.optimized);
    for (output, values) |*out_val, val| {
        const bits: u32 = @as(u32, val) << 16;
        out_val.* += @as(f32, @bitCast(bits)) * scale;
    }
}

// ---- Threaded matmul ----

/// Matmul dispatching on dtype, parallelized over output rows.
pub fn matmul(
    pool: ?*Pool,
    input: []const f32,
    w_bytes: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
    dtype: DataType,
) void {
    switch (dtype) {
        .FP32 => {
            const w: [*]const f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, w_bytes))).ptr;
            dispatchMatmul(f32, input, output, w, in_dim, out_dim, batch_size, pool);
        },
        .FP16 => {
            const w: [*]const f16 = @as([]const f16, @alignCast(std.mem.bytesAsSlice(f16, w_bytes))).ptr;
            dispatchMatmul(f16, input, output, w, in_dim, out_dim, batch_size, pool);
        },
        .BF16 => {
            const w: [*]const u16 = @as([]const u16, @alignCast(std.mem.bytesAsSlice(u16, w_bytes))).ptr;
            dispatchMatmul(u16, input, output, w, in_dim, out_dim, batch_size, pool);
        },
    }
}

/// Fused gate+up projections with SiLU*hadamard, parallelized.
/// Replaces: matmul(gate) + matmul(up) + siluHadamard(gate, up)
pub fn matmulSiluHadamard(
    pool: ?*Pool,
    input: []const f32,
    gate_w: []const u8,
    up_w: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
    dtype: DataType,
) void {
    switch (dtype) {
        .FP32 => {
            const gw: [*]const f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, gate_w))).ptr;
            const uw: [*]const f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, up_w))).ptr;
            dispatchFusedFFN(f32, input, output, gw, uw, in_dim, out_dim, batch_size, pool);
        },
        .FP16 => {
            const gw: [*]const f16 = @as([]const f16, @alignCast(std.mem.bytesAsSlice(f16, gate_w))).ptr;
            const uw: [*]const f16 = @as([]const f16, @alignCast(std.mem.bytesAsSlice(f16, up_w))).ptr;
            dispatchFusedFFN(f16, input, output, gw, uw, in_dim, out_dim, batch_size, pool);
        },
        .BF16 => {
            const gw: [*]const u16 = @as([]const u16, @alignCast(std.mem.bytesAsSlice(u16, gate_w))).ptr;
            const uw: [*]const u16 = @as([]const u16, @alignCast(std.mem.bytesAsSlice(u16, up_w))).ptr;
            dispatchFusedFFN(u16, input, output, gw, uw, in_dim, out_dim, batch_size, pool);
        },
    }
}

// ---- Internal: threaded dispatch ----

fn dispatchMatmul(
    comptime W: type,
    batch_in: []const f32,
    batch_out: []f32,
    w: [*]const W,
    input_dim: usize,
    output_dim: usize,
    batch_size: usize,
    pool: ?*Pool,
) void {
    const Kernel = struct {
        fn run(
            batch_input: []const f32,
            batch_output: []f32,
            weights: [*]const W,
            in_dim: usize,
            out_dim: usize,
            batch: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            for (start_row..end_row) |row| {
                const w_row = weights + row * in_dim;
                for (0..batch) |token| {
                    batch_output[token * out_dim + row] = dotW(W, batch_input[token * in_dim ..].ptr, w_row, in_dim);
                }
            }
        }
    };

    if (pool) |p| {
        if (output_dim >= 32) {
            const num_threads = p.threads.len + 1;
            const base = output_dim / num_threads;
            const extra = output_dim % num_threads;
            var wg: WaitGroup = .{};
            var start: usize = 0;
            for (0..num_threads) |thread_index| {
                const count = base + @intFromBool(thread_index < extra);
                p.spawnWg(&wg, Kernel.run, .{ batch_in, batch_out, w, input_dim, output_dim, batch_size, start, start + count });
                start += count;
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(batch_in, batch_out, w, input_dim, output_dim, batch_size, 0, output_dim);
}

fn dispatchFusedFFN(
    comptime W: type,
    batch_in: []const f32,
    batch_out: []f32,
    gate_w: [*]const W,
    up_w: [*]const W,
    input_dim: usize,
    output_dim: usize,
    batch_size: usize,
    pool: ?*Pool,
) void {
    const Kernel = struct {
        fn run(
            batch_input: []const f32,
            batch_output: []f32,
            gate_weights: [*]const W,
            up_weights: [*]const W,
            in_dim: usize,
            out_dim: usize,
            batch: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            @setFloatMode(.optimized);
            for (start_row..end_row) |row| {
                const w_off = row * in_dim;
                for (0..batch) |token| {
                    const input = batch_input[token * in_dim ..].ptr;
                    const gate_val = dotW(W, input, gate_weights + w_off, in_dim);
                    const up_val = dotW(W, input, up_weights + w_off, in_dim);
                    const silu_val = common.silu(gate_val);
                    batch_output[token * out_dim + row] = silu_val * up_val;
                }
            }
        }
    };

    if (pool) |p| {
        if (output_dim >= 32) {
            const num_threads = p.threads.len + 1;
            const base = output_dim / num_threads;
            const extra = output_dim % num_threads;
            var wg: WaitGroup = .{};
            var start: usize = 0;
            for (0..num_threads) |thread_index| {
                const count = base + @intFromBool(thread_index < extra);
                p.spawnWg(&wg, Kernel.run, .{ batch_in, batch_out, gate_w, up_w, input_dim, output_dim, batch_size, start, start + count });
                start += count;
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(batch_in, batch_out, gate_w, up_w, input_dim, output_dim, batch_size, 0, output_dim);
}

// ---- Generic dot product ----

inline fn dotW(comptime W: type, noalias a: [*]const f32, noalias b: [*]const W, len: usize) f32 {
    // f64 accumulator — inner loop of every BF16 matmul. Costs ~2-3×
    // prefill speed vs f32 with auto-vectorized FMA, but matches
    // llama.cpp's effective precision (their SIMD-parallel f32 sum still
    // accumulates noise; f64 is unambiguously correct).
    var sum: f64 = 0.0;
    for (0..len) |i| {
        sum += @as(f64, a[i]) * @as(f64, toF32(W, b[i]));
    }
    return @floatCast(sum);
}

inline fn toF32(comptime W: type, val: W) f32 {
    return switch (W) {
        f32 => val,
        f16 => @as(f32, @floatCast(val)),
        u16 => @as(f32, @bitCast(@as(u32, val) << 16)),
        else => unreachable,
    };
}
