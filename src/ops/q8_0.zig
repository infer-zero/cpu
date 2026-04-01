const std = @import("std");
const builtin = @import("builtin");
const common = @import("common.zig");

const Pool = std.Thread.Pool;
const WaitGroup = std.Thread.WaitGroup;

/// Number of R4 groups per output tile for batch matmul.
/// 16 groups = 64 output rows.
const tile_groups: usize = 16;

/// True when targeting a CPU with 512-bit VNNI (AVX-512 VNNI).
/// Zen 4/5, Intel Sapphire Rapids+. Build with -Dcpu=native or -Dcpu=znver4.
const has_avx512_vnni = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vnni);

// ---- Quantization ----

/// Quantize f32 values to Q8 (i8 with per-block f32 scales).
/// Blocks of 32 values. q_vals and q_scales must be pre-allocated.
pub fn quantizeF32ToQ8(
    input: []const f32,
    q_vals: []i8,
    q_scales: []f32,
) void {
    @setFloatMode(.optimized);
    const block_size = 32;
    const num_blocks = input.len / block_size;
    for (0..num_blocks) |block| {
        const src = input[block * block_size ..][0..block_size];
        const dest = q_vals[block * block_size ..][0..block_size];
        var max_abs: f32 = 0;
        for (src) |value| {
            const abs_val = @abs(value);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        if (max_abs == 0) {
            @memset(dest, 0);
            q_scales[block] = 0;
            continue;
        }
        const inv_scale = 127.0 / max_abs;
        q_scales[block] = max_abs / 127.0;
        for (src, dest) |value, *quantized| {
            quantized.* = @intFromFloat(@round(value * inv_scale));
        }
    }
}

// ---- Threaded matmul ----

/// Matmul: quantizes f32 inputs, then dispatches integer dot product.
/// q_vals/q_scales must be pre-allocated with at least batch_size * in_dim / (in_dim/32) elements.
pub fn matmul(
    pool: ?*Pool,
    input: []const f32,
    w_bytes: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
    q_vals: []i8,
    q_scales: []f32,
) void {
    const num_blocks = in_dim / 32;
    for (0..batch_size) |token| {
        quantizeF32ToQ8(
            input[token * in_dim ..][0..in_dim],
            q_vals[token * in_dim ..][0..in_dim],
            q_scales[token * num_blocks ..][0..num_blocks],
        );
    }
    matmulQ8(
        pool,
        q_vals,
        q_scales,
        w_bytes,
        output,
        in_dim,
        out_dim,
        batch_size,
    );
}

/// Matmul with pre-quantized i8 input, parallelized over output rows.
/// Use when the same quantized input is reused across multiple projections.
pub fn matmulQ8(
    pool: ?*Pool,
    q_vals: []const i8,
    q_scales: []const f32,
    w_bytes: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
) void {
    const num_blocks = in_dim / 32;

    const Kernel = struct {
        fn run(
            quantized_vals: []const i8,
            quantized_scales: []const f32,
            out: []f32,
            weights: []const u8,
            input_dim: usize,
            output_dim: usize,
            batch: usize,
            num_blocks_inner: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            const row_bytes = num_blocks_inner * 34;
            for (start_row..end_row) |row| {
                const weight_row = weights.ptr + row * row_bytes;
                for (0..batch) |token| {
                    out[token * output_dim + row] = dotQ8_0_q8(
                        quantized_vals[token * input_dim ..].ptr,
                        quantized_scales[token * num_blocks_inner ..].ptr,
                        weight_row,
                        num_blocks_inner,
                    );
                }
            }
        }
    };

    if (pool) |p| {
        if (out_dim >= 32) {
            const num_threads = p.threads.len + 1;
            const base = out_dim / num_threads;
            const extra = out_dim % num_threads;
            var wg: WaitGroup = .{};
            var start: usize = 0;
            for (0..num_threads) |thread_index| {
                const count = base + @intFromBool(thread_index < extra);
                p.spawnWg(
                    &wg,
                    Kernel.run,
                    .{
                        q_vals,
                        q_scales,
                        output,
                        w_bytes,
                        in_dim,
                        out_dim,
                        batch_size,
                        num_blocks,
                        start,
                        start + count,
                    },
                );
                start += count;
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(
        q_vals,
        q_scales,
        output,
        w_bytes,
        in_dim,
        out_dim,
        batch_size,
        num_blocks,
        0,
        out_dim,
    );
}

/// Fused gate+up projections with SiLU*hadamard, parallelized.
/// Quantizes f32 inputs, then dispatches integer dot product.
pub fn matmulSiluHadamard(
    pool: ?*Pool,
    input: []const f32,
    gate_weights: []const u8,
    up_weights: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
    q_vals: []i8,
    q_scales: []f32,
) void {
    const num_blocks = in_dim / 32;
    for (0..batch_size) |token| {
        quantizeF32ToQ8(
            input[token * in_dim ..][0..in_dim],
            q_vals[token * in_dim ..][0..in_dim],
            q_scales[token * num_blocks ..][0..num_blocks],
        );
    }
    matmulSiluHadamardQ8(
        pool,
        q_vals,
        q_scales,
        gate_weights,
        up_weights,
        output,
        in_dim,
        out_dim,
        batch_size,
    );
}

/// Fused gate+up with SiLU*hadamard using pre-quantized i8 input.
fn matmulSiluHadamardQ8(
    pool: ?*Pool,
    q_vals: []const i8,
    q_scales: []const f32,
    gate_weights: []const u8,
    up_weights: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
) void {
    const num_blocks = in_dim / 32;

    const Kernel = struct {
        fn run(
            quantized_vals: []const i8,
            quantized_scales: []const f32,
            out: []f32,
            gate_w: []const u8,
            up_w: []const u8,
            input_dim: usize,
            output_dim: usize,
            batch: usize,
            num_blocks_inner: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            @setFloatMode(.optimized);
            const row_bytes = num_blocks_inner * 34;
            for (start_row..end_row) |row| {
                const gate_row = gate_w.ptr + row * row_bytes;
                const up_row = up_w.ptr + row * row_bytes;
                for (0..batch) |token| {
                    const input_vals = quantized_vals[token * input_dim ..].ptr;
                    const input_scales = quantized_scales[token * num_blocks_inner ..].ptr;
                    const gate = dotQ8_0_q8(input_vals, input_scales, gate_row, num_blocks_inner);
                    const up = dotQ8_0_q8(input_vals, input_scales, up_row, num_blocks_inner);
                    out[token * output_dim + row] = common.silu(gate) * up;
                }
            }
        }
    };

    if (pool) |p| {
        if (out_dim >= 32) {
            const num_threads = p.threads.len + 1;
            const base = out_dim / num_threads;
            const extra = out_dim % num_threads;
            var wg: WaitGroup = .{};
            var start: usize = 0;
            for (0..num_threads) |thread_index| {
                const count = base + @intFromBool(thread_index < extra);
                p.spawnWg(
                    &wg,
                    Kernel.run,
                    .{
                        q_vals,
                        q_scales,
                        output,
                        gate_weights,
                        up_weights,
                        in_dim,
                        out_dim,
                        batch_size,
                        num_blocks,
                        start,
                        start + count,
                    },
                );
                start += count;
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(
        q_vals,
        q_scales,
        output,
        gate_weights,
        up_weights,
        in_dim,
        out_dim,
        batch_size,
        num_blocks,
        0,
        out_dim,
    );
}

// ---- R4 matmul (4-row interleaved) ----

/// Matmul with R4-interleaved weights: quantizes f32 input, then dispatches R4 kernel.
pub fn matmulR4(
    pool: ?*Pool,
    input: []const f32,
    w_bytes: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
    q_vals: []i8,
    q_scales: []f32,
) void {
    const num_blocks = in_dim / 32;
    for (0..batch_size) |token| {
        quantizeF32ToQ8(
            input[token * in_dim ..][0..in_dim],
            q_vals[token * in_dim ..][0..in_dim],
            q_scales[token * num_blocks ..][0..num_blocks],
        );
    }
    matmulQ8R4(
        pool,
        q_vals,
        q_scales,
        w_bytes,
        output,
        in_dim,
        out_dim,
        batch_size,
    );
}

/// Matmul with pre-quantized i8 input and R4-interleaved Q8_0 weights.
/// Groups of 4 rows are interleaved at block granularity for better cache utilization.
/// Uses tile-based thread distribution for batch>1 and multi-token kernel for 2-token ILP.
pub fn matmulQ8R4(
    pool: ?*Pool,
    q_vals: []const i8,
    q_scales: []const f32,
    w_bytes: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
) void {
    const num_blocks = in_dim / 32;
    const num_groups = (out_dim + 3) / 4;
    const effective_tile = if (batch_size > 1) tile_groups else 1;
    const num_tiles = (num_groups + effective_tile - 1) / effective_tile;

    const Kernel = struct {
        fn run(
            quantized_vals: []const i8,
            quantized_scales: []const f32,
            out: []f32,
            weights: []const u8,
            input_dim: usize,
            output_dim: usize,
            batch: usize,
            nblocks: usize,
            start_tile: usize,
            end_tile: usize,
            t_groups: usize,
            total_groups: usize,
        ) void {
            for (start_tile..end_tile) |tile| {
                const g_start = tile * t_groups;
                const g_end = @min(g_start + t_groups, total_groups);
                for (g_start..g_end) |group| {
                    const base_row = group * 4;
                    const group_ptr = weights.ptr + group * nblocks * (4 * 34);

                    var token: usize = 0;
                    while (token + 1 < batch) : (token += 2) {
                        const results = dotQ8_0_q8_R4x2(
                            quantized_vals[token * input_dim ..].ptr,
                            quantized_scales[token * nblocks ..].ptr,
                            quantized_vals[(token + 1) * input_dim ..].ptr,
                            quantized_scales[(token + 1) * nblocks ..].ptr,
                            group_ptr,
                            nblocks,
                        );
                        inline for (0..4) |r| {
                            if (base_row + r < output_dim) {
                                out[token * output_dim + base_row + r] = results[0][r];
                                out[(token + 1) * output_dim + base_row + r] = results[1][r];
                            }
                        }
                    }
                    if (token < batch) {
                        const results = dotQ8_0_q8_R4(
                            quantized_vals[token * input_dim ..].ptr,
                            quantized_scales[token * nblocks ..].ptr,
                            group_ptr,
                            nblocks,
                        );
                        inline for (0..4) |r| {
                            if (base_row + r < output_dim) {
                                out[token * output_dim + base_row + r] = results[r];
                            }
                        }
                    }
                }
            }
        }
    };

    if (pool) |p| {
        if (num_tiles >= 2) {
            const num_threads = p.threads.len + 1;
            const base = num_tiles / num_threads;
            const extra = num_tiles % num_threads;
            var wg: WaitGroup = .{};
            var start: usize = 0;
            for (0..num_threads) |thread_index| {
                const count = base + @intFromBool(thread_index < extra);
                if (count > 0) {
                    p.spawnWg(
                        &wg,
                        Kernel.run,
                        .{
                            q_vals,
                            q_scales,
                            output,
                            w_bytes,
                            in_dim,
                            out_dim,
                            batch_size,
                            num_blocks,
                            start,
                            start + count,
                            effective_tile,
                            num_groups,
                        },
                    );
                    start += count;
                }
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(
        q_vals,
        q_scales,
        output,
        w_bytes,
        in_dim,
        out_dim,
        batch_size,
        num_blocks,
        0,
        num_tiles,
        effective_tile,
        num_groups,
    );
}

/// Fused gate+up projections with SiLU*hadamard using R4-interleaved weights.
pub fn matmulSiluHadamardR4(
    pool: ?*Pool,
    input: []const f32,
    gate_weights: []const u8,
    up_weights: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
    q_vals: []i8,
    q_scales: []f32,
) void {
    const num_blocks = in_dim / 32;
    for (0..batch_size) |token| {
        quantizeF32ToQ8(
            input[token * in_dim ..][0..in_dim],
            q_vals[token * in_dim ..][0..in_dim],
            q_scales[token * num_blocks ..][0..num_blocks],
        );
    }
    matmulSiluHadamardQ8R4(
        pool,
        q_vals,
        q_scales,
        gate_weights,
        up_weights,
        output,
        in_dim,
        out_dim,
        batch_size,
    );
}

/// Fused gate+up with SiLU*hadamard using pre-quantized i8 input and R4-interleaved weights.
/// Uses tile-based thread distribution for batch>1 and multi-token kernel for 2-token ILP.
fn matmulSiluHadamardQ8R4(
    pool: ?*Pool,
    q_vals: []const i8,
    q_scales: []const f32,
    gate_weights: []const u8,
    up_weights: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
) void {
    const num_blocks = in_dim / 32;
    const num_groups = (out_dim + 3) / 4;
    const effective_tile = if (batch_size > 1) tile_groups else 1;
    const num_tiles = (num_groups + effective_tile - 1) / effective_tile;

    const Kernel = struct {
        fn run(
            quantized_vals: []const i8,
            quantized_scales: []const f32,
            out: []f32,
            gate_w: []const u8,
            up_w: []const u8,
            input_dim: usize,
            output_dim: usize,
            batch: usize,
            nblocks: usize,
            start_tile: usize,
            end_tile: usize,
            t_groups: usize,
            total_groups: usize,
        ) void {
            @setFloatMode(.optimized);
            for (start_tile..end_tile) |tile| {
                const g_start = tile * t_groups;
                const g_end = @min(g_start + t_groups, total_groups);
                for (g_start..g_end) |group| {
                    const base_row = group * 4;
                    const gate_group_ptr = gate_w.ptr + group * nblocks * (4 * 34);
                    const up_group_ptr = up_w.ptr + group * nblocks * (4 * 34);

                    var token: usize = 0;
                    while (token + 1 < batch) : (token += 2) {
                        const vals_0 = quantized_vals[token * input_dim ..].ptr;
                        const scales_0 = quantized_scales[token * nblocks ..].ptr;
                        const vals_1 = quantized_vals[(token + 1) * input_dim ..].ptr;
                        const scales_1 = quantized_scales[(token + 1) * nblocks ..].ptr;
                        const gate_results = dotQ8_0_q8_R4x2(vals_0, scales_0, vals_1, scales_1, gate_group_ptr, nblocks);
                        const up_results = dotQ8_0_q8_R4x2(vals_0, scales_0, vals_1, scales_1, up_group_ptr, nblocks);
                        inline for (0..2) |t| {
                            inline for (0..4) |r| {
                                if (base_row + r < output_dim) {
                                    const gate = gate_results[t][r];
                                    out[(token + t) * output_dim + base_row + r] = common.silu(gate) * up_results[t][r];
                                }
                            }
                        }
                    }
                    if (token < batch) {
                        const input_vals = quantized_vals[token * input_dim ..].ptr;
                        const input_scales = quantized_scales[token * nblocks ..].ptr;
                        const gate_results = dotQ8_0_q8_R4(input_vals, input_scales, gate_group_ptr, nblocks);
                        const up_results = dotQ8_0_q8_R4(input_vals, input_scales, up_group_ptr, nblocks);
                        inline for (0..4) |r| {
                            if (base_row + r < output_dim) {
                                const gate = gate_results[r];
                                out[token * output_dim + base_row + r] = common.silu(gate) * up_results[r];
                            }
                        }
                    }
                }
            }
        }
    };

    if (pool) |p| {
        if (num_tiles >= 2) {
            const num_threads = p.threads.len + 1;
            const base = num_tiles / num_threads;
            const extra = num_tiles % num_threads;
            var wg: WaitGroup = .{};
            var start: usize = 0;
            for (0..num_threads) |thread_index| {
                const count = base + @intFromBool(thread_index < extra);
                if (count > 0) {
                    p.spawnWg(
                        &wg,
                        Kernel.run,
                        .{
                            q_vals,
                            q_scales,
                            output,
                            gate_weights,
                            up_weights,
                            in_dim,
                            out_dim,
                            batch_size,
                            num_blocks,
                            start,
                            start + count,
                            effective_tile,
                            num_groups,
                        },
                    );
                    start += count;
                }
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(
        q_vals,
        q_scales,
        output,
        gate_weights,
        up_weights,
        in_dim,
        out_dim,
        batch_size,
        num_blocks,
        0,
        num_tiles,
        effective_tile,
        num_groups,
    );
}

// ---- Dot product kernels ----

/// Integer dot product: pre-quantized i8 input x Q8_0 weights.
/// Uses the sign trick with AVX_VNNI vpdpbusd (unsigned x signed byte dot product):
///   abs(x) provides the unsigned operand, sign(x)*w provides the signed operand,
///   so vpdpbusd(0, |x|, sign(x)*w) = sum(x_i * w_i) without needing a correction term.
/// Manual 2x unrolling since inline asm prevents LLVM's loop unroller.
inline fn dotQ8_0_q8(
    noalias x_vals: [*]const i8,
    noalias x_scales: [*]const f32,
    noalias w_bytes: [*]const u8,
    num_blocks: usize,
) f32 {
    @setFloatMode(.optimized);
    const V8f = @Vector(8, f32);
    const V32i8 = @Vector(32, i8);
    const zero_i32: @Vector(8, i32) = @splat(0);
    var acc: V8f = @splat(0);

    var block: usize = 0;
    if (comptime has_avx512_vnni) {
        const V64i8 = @Vector(64, i8);
        const zero_i32_16: @Vector(16, i32) = @splat(0);

        // 2-block-at-a-time loop using 512-bit vpdpbusd
        while (block + 1 < num_blocks) : (block += 2) {
            // Input: 64 contiguous bytes (2 blocks)
            const input_64: V64i8 = (x_vals + block * 32)[0..64].*;

            // Weights: load 2 separate 32-byte chunks (34-byte stride), concatenate
            const bp0 = w_bytes + block * 34;
            const bp1 = w_bytes + (block + 1) * 34;
            const ws0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[0..2], .little))));
            const ws1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[0..2], .little))));
            const w0: [*]const i8 = @ptrCast(bp0 + 2);
            const w1: [*]const i8 = @ptrCast(bp1 + 2);
            var w_arr: [64]i8 = undefined;
            w_arr[0..32].* = w0[0..32].*;
            w_arr[32..64].* = w1[0..32].*;
            const weight_64: V64i8 = w_arr;

            // 512-bit dot product → 16 i32 results
            const dots = intDotI8_512(zero_i32_16, input_64, weight_64);
            const dots_arr: [16]i32 = dots;

            // Split and apply per-block scales
            const scale0: V8f = @splat(x_scales[block] * ws0);
            const scale1: V8f = @splat(x_scales[block + 1] * ws1);
            acc += @as(V8f, @floatFromInt(@as(@Vector(8, i32), dots_arr[0..8].*))) * scale0;
            acc += @as(V8f, @floatFromInt(@as(@Vector(8, i32), dots_arr[8..16].*))) * scale1;
        }
    } else {
        // 4x block unrolling
        while (block + 3 < num_blocks) : (block += 4) {
            inline for (0..4) |i| {
                const block_ptr = w_bytes + (block + i) * 34;
                const weight_scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
                const weight_data: [*]const i8 = @ptrCast(block_ptr + 2);
                const input_data = x_vals + (block + i) * 32;
                const combined_scale: V8f = @splat(x_scales[block + i] * weight_scale);

                const input_i8: V32i8 = input_data[0..32].*;
                const weight_i8: V32i8 = weight_data[0..32].*;

                acc += @as(V8f, @floatFromInt(intDotI8(zero_i32, input_i8, weight_i8))) * combined_scale;
            }
        }
        // 2x step-down
        while (block + 1 < num_blocks) : (block += 2) {
            inline for (0..2) |i| {
                const block_ptr = w_bytes + (block + i) * 34;
                const weight_scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
                const weight_data: [*]const i8 = @ptrCast(block_ptr + 2);
                const input_data = x_vals + (block + i) * 32;
                const combined_scale: V8f = @splat(x_scales[block + i] * weight_scale);

                const input_i8: V32i8 = input_data[0..32].*;
                const weight_i8: V32i8 = weight_data[0..32].*;

                acc += @as(V8f, @floatFromInt(intDotI8(zero_i32, input_i8, weight_i8))) * combined_scale;
            }
        }
    }
    // Single block remainder
    if (block < num_blocks) {
        const block_ptr = w_bytes + block * 34;
        const weight_scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
        const weight_data: [*]const i8 = @ptrCast(block_ptr + 2);
        const input_data = x_vals + block * 32;
        const combined_scale: V8f = @splat(x_scales[block] * weight_scale);

        const input_i8: V32i8 = input_data[0..32].*;
        const weight_i8: V32i8 = weight_data[0..32].*;

        acc += @as(V8f, @floatFromInt(intDotI8(zero_i32, input_i8, weight_i8))) * combined_scale;
    }

    return @reduce(.Add, acc);
}

/// 4-row dot product kernel for R4-interleaved Q8_0 weights.
/// Returns 4 dot products (one per row in the group) simultaneously.
/// Input blocks are loaded once and reused across all 4 weight rows.
inline fn dotQ8_0_q8_R4(
    noalias x_vals: [*]const i8,
    noalias x_scales: [*]const f32,
    noalias group_ptr: [*]const u8,
    num_blocks: usize,
) [4]f32 {
    @setFloatMode(.optimized);
    const V8f = @Vector(8, f32);
    const V32i8 = @Vector(32, i8);
    const zero_i32: @Vector(8, i32) = @splat(0);
    const zero_v8f: V8f = @splat(0);
    var acc: [4]V8f = .{ zero_v8f, zero_v8f, zero_v8f, zero_v8f };

    var block: usize = 0;
    if (comptime has_avx512_vnni) {
        const V64i8 = @Vector(64, i8);
        const zero_i32_16: @Vector(16, i32) = @splat(0);

        // 2-block-at-a-time loop using 512-bit vpdpbusd
        while (block + 1 < num_blocks) : (block += 2) {
            const input_64: V64i8 = (x_vals + block * 32)[0..64].*;
            const is0 = x_scales[block];
            const is1 = x_scales[block + 1];

            inline for (0..4) |row| {
                const bp0 = group_ptr + block * (4 * 34) + row * 34;
                const bp1 = group_ptr + (block + 1) * (4 * 34) + row * 34;
                const ws0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[0..2], .little))));
                const ws1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[0..2], .little))));
                var w_arr: [64]i8 = undefined;
                w_arr[0..32].* = @as([*]const i8, @ptrCast(bp0 + 2))[0..32].*;
                w_arr[32..64].* = @as([*]const i8, @ptrCast(bp1 + 2))[0..32].*;

                const dots = intDotI8_512(zero_i32_16, input_64, @as(V64i8, w_arr));
                const dots_arr: [16]i32 = dots;
                acc[row] += @as(V8f, @floatFromInt(@as(@Vector(8, i32), dots_arr[0..8].*))) * @as(V8f, @splat(is0 * ws0));
                acc[row] += @as(V8f, @floatFromInt(@as(@Vector(8, i32), dots_arr[8..16].*))) * @as(V8f, @splat(is1 * ws1));
            }
        }
    } else {
        // 2x block unrolling
        while (block + 1 < num_blocks) : (block += 2) {
            inline for (0..2) |i| {
                const b = block + i;
                const input_data = x_vals + b * 32;
                const input_i8: V32i8 = input_data[0..32].*;
                const input_scale = x_scales[b];

                inline for (0..4) |row| {
                    const block_ptr = group_ptr + b * (4 * 34) + row * 34;
                    const weight_scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
                    const weight_data: [*]const i8 = @ptrCast(block_ptr + 2);
                    const weight_i8: V32i8 = weight_data[0..32].*;
                    const combined_scale: V8f = @splat(input_scale * weight_scale);
                    acc[row] += @as(V8f, @floatFromInt(intDotI8(zero_i32, input_i8, weight_i8))) * combined_scale;
                }
            }
        }
    }

    // Handle odd last block
    if (block < num_blocks) {
        const input_data = x_vals + block * 32;
        const input_i8: V32i8 = input_data[0..32].*;
        const input_scale = x_scales[block];

        inline for (0..4) |row| {
            const block_ptr = group_ptr + block * (4 * 34) + row * 34;
            const weight_scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
            const weight_data: [*]const i8 = @ptrCast(block_ptr + 2);
            const weight_i8: V32i8 = weight_data[0..32].*;
            const combined_scale: V8f = @splat(input_scale * weight_scale);
            acc[row] += @as(V8f, @floatFromInt(intDotI8(zero_i32, input_i8, weight_i8))) * combined_scale;
        }
    }

    return .{
        @reduce(.Add, acc[0]),
        @reduce(.Add, acc[1]),
        @reduce(.Add, acc[2]),
        @reduce(.Add, acc[3]),
    };
}

/// 2-token × 4-row dot product kernel for R4-interleaved Q8_0 weights.
/// Processes 2 tokens simultaneously against the same weight group, amortizing weight loads.
/// No block unrolling (8 V8f accumulators = 8 of 16 YMM registers).
inline fn dotQ8_0_q8_R4x2(
    noalias x_vals_0: [*]const i8,
    noalias x_scales_0: [*]const f32,
    noalias x_vals_1: [*]const i8,
    noalias x_scales_1: [*]const f32,
    noalias group_ptr: [*]const u8,
    num_blocks: usize,
) [2][4]f32 {
    @setFloatMode(.optimized);
    const V8f = @Vector(8, f32);
    const V32i8 = @Vector(32, i8);
    const zero_i32: @Vector(8, i32) = @splat(0);
    const zero_v8f: V8f = @splat(0);
    var acc0: [4]V8f = .{ zero_v8f, zero_v8f, zero_v8f, zero_v8f };
    var acc1: [4]V8f = .{ zero_v8f, zero_v8f, zero_v8f, zero_v8f };

    var block: usize = 0;
    if (comptime has_avx512_vnni) {
        const V64i8 = @Vector(64, i8);
        const zero_i32_16: @Vector(16, i32) = @splat(0);

        // 2-block-at-a-time loop using 512-bit vpdpbusd
        while (block + 1 < num_blocks) : (block += 2) {
            const input0_64: V64i8 = (x_vals_0 + block * 32)[0..64].*;
            const input1_64: V64i8 = (x_vals_1 + block * 32)[0..64].*;
            const is0_0 = x_scales_0[block];
            const is0_1 = x_scales_0[block + 1];
            const is1_0 = x_scales_1[block];
            const is1_1 = x_scales_1[block + 1];

            inline for (0..4) |row| {
                const bp0 = group_ptr + block * (4 * 34) + row * 34;
                const bp1 = group_ptr + (block + 1) * (4 * 34) + row * 34;
                const ws0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[0..2], .little))));
                const ws1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[0..2], .little))));
                var w_arr: [64]i8 = undefined;
                w_arr[0..32].* = @as([*]const i8, @ptrCast(bp0 + 2))[0..32].*;
                w_arr[32..64].* = @as([*]const i8, @ptrCast(bp1 + 2))[0..32].*;
                const weight_64: V64i8 = w_arr;

                // Token 0
                const dots0 = intDotI8_512(zero_i32_16, input0_64, weight_64);
                const dots0_arr: [16]i32 = dots0;
                acc0[row] += @as(V8f, @floatFromInt(@as(@Vector(8, i32), dots0_arr[0..8].*))) * @as(V8f, @splat(is0_0 * ws0));
                acc0[row] += @as(V8f, @floatFromInt(@as(@Vector(8, i32), dots0_arr[8..16].*))) * @as(V8f, @splat(is0_1 * ws1));

                // Token 1
                const dots1 = intDotI8_512(zero_i32_16, input1_64, weight_64);
                const dots1_arr: [16]i32 = dots1;
                acc1[row] += @as(V8f, @floatFromInt(@as(@Vector(8, i32), dots1_arr[0..8].*))) * @as(V8f, @splat(is1_0 * ws0));
                acc1[row] += @as(V8f, @floatFromInt(@as(@Vector(8, i32), dots1_arr[8..16].*))) * @as(V8f, @splat(is1_1 * ws1));
            }
        }
    }

    // Remaining blocks (all blocks if no AVX-512, or odd remainder)
    while (block < num_blocks) : (block += 1) {
        // Token 0 input
        const input0: V32i8 = (x_vals_0 + block * 32)[0..32].*;
        const scale0 = x_scales_0[block];

        // Token 1 input
        const input1: V32i8 = (x_vals_1 + block * 32)[0..32].*;
        const scale1 = x_scales_1[block];

        inline for (0..4) |row| {
            const block_ptr = group_ptr + block * (4 * 34) + row * 34;
            const weight_scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
            const weight_data: [*]const i8 = @ptrCast(block_ptr + 2);
            const weight_i8: V32i8 = weight_data[0..32].*;

            const combined_scale0: V8f = @splat(scale0 * weight_scale);
            acc0[row] += @as(V8f, @floatFromInt(intDotI8(zero_i32, input0, weight_i8))) * combined_scale0;

            const combined_scale1: V8f = @splat(scale1 * weight_scale);
            acc1[row] += @as(V8f, @floatFromInt(intDotI8(zero_i32, input1, weight_i8))) * combined_scale1;
        }
    }

    return .{
        .{ @reduce(.Add, acc0[0]), @reduce(.Add, acc0[1]), @reduce(.Add, acc0[2]), @reduce(.Add, acc0[3]) },
        .{ @reduce(.Add, acc1[0]), @reduce(.Add, acc1[1]), @reduce(.Add, acc1[2]), @reduce(.Add, acc1[3]) },
    };
}

/// Signed i8 × signed i8 dot product with i32 accumulation.
/// result[i] = acc[i] + sum(a[4i+j] * b[4i+j]) for j=0..3.
/// ARM with dotprod: uses sdot directly (signed × signed, no sign trick needed).
/// x86/fallback: applies sign trick internally, then delegates to vpdpbusd.
inline fn intDotI8(
    acc: @Vector(8, i32),
    a: @Vector(32, i8),
    b: @Vector(32, i8),
) @Vector(8, i32) {
    @setEvalBranchQuota(10000);
    if (builtin.mode == .Debug) {
        var dots: [8]i32 = acc;
        inline for (0..8) |i| {
            inline for (0..4) |j| {
                const idx = i * 4 + j;
                dots[i] += @as(i32, a[idx]) * @as(i32, b[idx]);
            }
        }
        return dots;
    }

    if (comptime builtin.cpu.arch == .aarch64) {
        if (comptime std.Target.aarch64.featureSetHas(builtin.cpu.features, .dotprod)) {
            const a_arr: [32]i8 = a;
            const b_arr: [32]i8 = b;
            const acc_arr: [8]i32 = acc;

            const res_lo = asm ("sdot %[out].4s, %[a].16b, %[b].16b"
                : [out] "=w" (-> @Vector(4, i32)),
                : [a] "w" (@as(@Vector(16, i8), a_arr[0..16].*)),
                  [b] "w" (@as(@Vector(16, i8), b_arr[0..16].*)),
                  [tied] "0" (@as(@Vector(4, i32), acc_arr[0..4].*)),
            );
            const res_hi = asm ("sdot %[out].4s, %[a].16b, %[b].16b"
                : [out] "=w" (-> @Vector(4, i32)),
                : [a] "w" (@as(@Vector(16, i8), a_arr[16..32].*)),
                  [b] "w" (@as(@Vector(16, i8), b_arr[16..32].*)),
                  [tied] "0" (@as(@Vector(4, i32), acc_arr[4..8].*)),
            );

            var result: [8]i32 = undefined;
            result[0..4].* = res_lo;
            result[4..8].* = res_hi;
            return result;
        }
    }

    // x86 / fallback: apply sign trick, then use vpdpbusd.
    const zero: @Vector(32, i8) = @splat(0);
    const abs_a = @abs(a);
    const a_negative = a < zero;
    const signed_b: @Vector(32, u8) = @bitCast(@select(i8, a_negative, zero -% b, b));
    return vpdpbusd(acc, abs_a, signed_b);
}

/// Unsigned x signed byte dot product with i32 accumulation.
/// result[i] = acc[i] + sum(unsigned[4i+j] * signed[4i+j]) for j=0..3.
inline fn vpdpbusd(
    acc: @Vector(8, i32),
    unsigned: @Vector(32, u8),
    signed: @Vector(32, u8),
) @Vector(8, i32) {
    if (builtin.mode == .Debug) {
        // Scalar reference for debugging.
        const s: @Vector(32, i8) = @bitCast(signed);
        var dots: [8]i32 = acc;
        inline for (0..8) |i| {
            inline for (0..4) |j| {
                const idx = i * 4 + j;
                dots[i] += @as(i32, @intCast(unsigned[idx])) * @as(i32, s[idx]);
            }
        }
        return dots;
    }

    if (comptime builtin.cpu.arch == .x86_64) {
        return asm ("{vex} vpdpbusd %[b], %[a], %[out]"
            : [out] "=x" (-> @Vector(8, i32)),
            : [a] "x" (unsigned),
              [b] "x" (signed),
              [tied] "0" (acc),
        );
    }

    if (comptime builtin.cpu.arch == .aarch64) {
        if (comptime std.Target.aarch64.featureSetHas(builtin.cpu.features, .dotprod)) {
            // NEON sdot: process 32 bytes as two 128-bit halves.
            // unsigned values are <= 127, so bitcast to i8 is safe for signed x signed sdot.
            const u: [32]u8 = unsigned;
            const s: [32]u8 = signed;
            const a: [8]i32 = acc;

            const res_lo = asm ("sdot %[out].4s, %[a].16b, %[b].16b"
                : [out] "=w" (-> @Vector(4, i32)),
                : [a] "w" (@as(@Vector(16, i8), @bitCast(u[0..16].*))),
                  [b] "w" (@as(@Vector(16, i8), @bitCast(s[0..16].*))),
                  [tied] "0" (@as(@Vector(4, i32), a[0..4].*)),
            );
            const res_hi = asm ("sdot %[out].4s, %[a].16b, %[b].16b"
                : [out] "=w" (-> @Vector(4, i32)),
                : [a] "w" (@as(@Vector(16, i8), @bitCast(u[16..32].*))),
                  [b] "w" (@as(@Vector(16, i8), @bitCast(s[16..32].*))),
                  [tied] "0" (@as(@Vector(4, i32), a[4..8].*)),
            );

            var result: [8]i32 = undefined;
            result[0..4].* = res_lo;
            result[4..8].* = res_hi;
            return result;
        }
    }

    // Scalar fallback for other architectures.
    const s: @Vector(32, i8) = @bitCast(signed);
    var dots: [8]i32 = acc;
    inline for (0..8) |i| {
        inline for (0..4) |j| {
            const idx = i * 4 + j;
            dots[i] += @as(i32, @intCast(unsigned[idx])) * @as(i32, s[idx]);
        }
    }
    return dots;
}

/// 512-bit unsigned x signed byte dot product with i32 accumulation (EVEX vpdpbusd on ZMM).
/// result[i] = acc[i] + sum(unsigned[4i+j] * signed[4i+j]) for j=0..3, i=0..15.
inline fn vpdpbusd512(
    acc: @Vector(16, i32),
    unsigned: @Vector(64, u8),
    signed: @Vector(64, u8),
) @Vector(16, i32) {
    return asm ("vpdpbusd %[b], %[a], %[out]"
        : [out] "=v" (-> @Vector(16, i32)),
        : [a] "v" (unsigned),
          [b] "v" (signed),
          [tied] "0" (acc),
    );
}

/// 512-bit signed i8 × signed i8 dot product with sign trick.
/// Processes 64 bytes (2 Q8_0 blocks) in a single instruction.
inline fn intDotI8_512(
    acc: @Vector(16, i32),
    a: @Vector(64, i8),
    b: @Vector(64, i8),
) @Vector(16, i32) {
    const zero: @Vector(64, i8) = @splat(0);
    const abs_a = @abs(a);
    const a_negative = a < zero;
    const signed_b: @Vector(64, u8) = @bitCast(@select(i8, a_negative, zero -% b, b));
    return vpdpbusd512(acc, abs_a, signed_b);
}

inline fn dotF32(noalias a: [*]const f32, noalias b: [*]const f32, len: usize) f32 {
    @setFloatMode(.optimized);
    var sum: f32 = 0.0;
    for (0..len) |i| sum += a[i] * b[i];
    return sum;
}
