const std = @import("std");
const builtin = @import("builtin");
const common = @import("common.zig");

const Pool = std.Thread.Pool;
const WaitGroup = std.Thread.WaitGroup;

/// Q4_K super-block: 256 elements in 144 bytes.
/// Layout: d[2](f16) + dmin[2](f16) + scales[12](6-bit packed) + qs[128](nibbles).
const Q4K_BLOCK_SIZE: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;
const Q4K_HEADER_BYTES: usize = 16; // d + dmin + scales
const Q4K_DATA_BYTES: usize = 128; // qs

/// True when targeting a CPU with 512-bit VNNI (AVX-512 VNNI).
const has_avx512_vnni = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vnni);

/// Max Q4_K blocks per row for dequant scratch buffers (covers up to dim=32768).
const max_num_super_blocks: usize = 128; // 128 * 256 = 32768 elements
/// Max tokens for gate accumulator in fused SiLU kernels.
const max_batch_size: usize = 4096;

// ---- Quantization (re-exports from q4_0) ----

pub const quantizeF32ToQ8 = @import("q4_0.zig").quantizeF32ToQ8;

// ---- Q4_K scale unpacking ----

/// Unpack 6-bit scales and mins from the 12-byte packed array in a Q4_K block header.
/// Returns 8 scales and 8 mins as u8 values.
/// Matches the reference in base/src/tensor.zig lines 216-234.
inline fn unpackScales(scales_raw: *const [12]u8) struct { scales: [8]u8, mins: [8]u8 } {
    return .{
        .scales = .{
            scales_raw[0] & 0x3F,
            scales_raw[1] & 0x3F,
            scales_raw[2] & 0x3F,
            scales_raw[3] & 0x3F,
            (scales_raw[8] & 0x0F) | ((scales_raw[0] >> 6) << 4),
            (scales_raw[9] & 0x0F) | ((scales_raw[1] >> 6) << 4),
            (scales_raw[10] & 0x0F) | ((scales_raw[2] >> 6) << 4),
            (scales_raw[11] & 0x0F) | ((scales_raw[3] >> 6) << 4),
        },
        .mins = .{
            scales_raw[4] & 0x3F,
            scales_raw[5] & 0x3F,
            scales_raw[6] & 0x3F,
            scales_raw[7] & 0x3F,
            (scales_raw[8] >> 4) | ((scales_raw[4] >> 6) << 4),
            (scales_raw[9] >> 4) | ((scales_raw[5] >> 6) << 4),
            (scales_raw[10] >> 4) | ((scales_raw[6] >> 6) << 4),
            (scales_raw[11] >> 4) | ((scales_raw[7] >> 6) << 4),
        },
    };
}

// ---- SIMD dot product: Q4_K weight × Q8 input ----

/// Dot product of Q4_K weight row × Q8-quantized input vector.
/// Each Q4_K super-block (256 elements) consumes 8 consecutive Q8 blocks (32 elements each).
///
/// Math per sub-block of 32 elements:
///   partial = x_scale * (d * scale_j * dot(x_q, nibble) - dmin * min_j * sum(x_q))
///
/// where dot(x_q, nibble) is the integer dot product and sum(x_q) is the
/// sum of Q8 values (needed for the dmin bias correction term).
inline fn dotQ4_K_q8(
    noalias x_vals: [*]const i8,
    noalias x_scales: [*]const f32,
    noalias w_bytes: [*]const u8,
    num_super_blocks: usize,
) f32 {
    @setFloatMode(.optimized);
    const V32i8 = @Vector(32, i8);
    const zero_i32: @Vector(8, i32) = @splat(0);
    const ones_i8: V32i8 = @splat(1);

    var acc: f32 = 0;

    for (0..num_super_blocks) |sb| {
        const block_ptr = w_bytes + sb * Q4K_BLOCK_BYTES;

        // Read header: d, dmin, scales
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
        const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[2..4], .little))));
        const sm = unpackScales(block_ptr[4..16]);

        const qs = block_ptr + Q4K_HEADER_BYTES;

        // Process 4 groups of 64 elements (2 sub-blocks of 32 each)
        for (0..4) |j| {
            const sc1: f32 = d * @as(f32, @floatFromInt(sm.scales[j * 2]));
            const m1: f32 = dmin * @as(f32, @floatFromInt(sm.mins[j * 2]));
            const sc2: f32 = d * @as(f32, @floatFromInt(sm.scales[j * 2 + 1]));
            const m2: f32 = dmin * @as(f32, @floatFromInt(sm.mins[j * 2 + 1]));

            // Load 32 qs bytes for this group, split into low/high nibbles
            const full_nibbles_raw: @Vector(32, u8) = (qs + j * 32)[0..32].*;
            const low_u8: @Vector(32, u8) = full_nibbles_raw & @as(@Vector(32, u8), @splat(0x0F));
            const high_u8: @Vector(32, u8) = full_nibbles_raw >> @as(@Vector(32, u8), @splat(4));

            // Q8 input sub-block 0 (low nibbles)
            const x_block_0 = sb * 8 + j * 2;
            const input_lo: V32i8 = (x_vals + x_block_0 * 32)[0..32].*;
            const x_scale_lo = x_scales[x_block_0];

            const dot_lo_sum: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, low_u8, input_lo)));
            const sum_lo_total: f32 = @floatFromInt(@reduce(.Add, intDotI8(zero_i32, input_lo, ones_i8)));
            acc += x_scale_lo * (sc1 * dot_lo_sum - m1 * sum_lo_total);

            // Q8 input sub-block 1 (high nibbles)
            const x_block_1 = sb * 8 + j * 2 + 1;
            const input_hi: V32i8 = (x_vals + x_block_1 * 32)[0..32].*;
            const x_scale_hi = x_scales[x_block_1];

            const dot_hi_sum: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, high_u8, input_hi)));
            const sum_hi_total: f32 = @floatFromInt(@reduce(.Add, intDotI8(zero_i32, input_hi, ones_i8)));
            acc += x_scale_hi * (sc2 * dot_hi_sum - m2 * sum_hi_total);
        }
    }

    return acc;
}

/// 2-token variant: unpack Q4_K weight block once, compute dot products for 2 tokens.
inline fn dotQ4_K_q8_x2(
    noalias x_vals_0: [*]const i8,
    noalias x_scales_0: [*]const f32,
    noalias x_vals_1: [*]const i8,
    noalias x_scales_1: [*]const f32,
    noalias w_bytes: [*]const u8,
    num_super_blocks: usize,
) [2]f32 {
    @setFloatMode(.optimized);
    const V32i8 = @Vector(32, i8);
    const zero_i32: @Vector(8, i32) = @splat(0);
    const ones_i8: V32i8 = @splat(1);

    var sum0: f32 = 0;
    var sum1: f32 = 0;

    for (0..num_super_blocks) |sb| {
        const block_ptr = w_bytes + sb * Q4K_BLOCK_BYTES;

        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
        const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[2..4], .little))));
        const sm = unpackScales(block_ptr[4..16]);

        const qs = block_ptr + Q4K_HEADER_BYTES;

        for (0..4) |j| {
            const sc1: f32 = d * @as(f32, @floatFromInt(sm.scales[j * 2]));
            const m1: f32 = dmin * @as(f32, @floatFromInt(sm.mins[j * 2]));
            const sc2: f32 = d * @as(f32, @floatFromInt(sm.scales[j * 2 + 1]));
            const m2: f32 = dmin * @as(f32, @floatFromInt(sm.mins[j * 2 + 1]));

            const full_nibbles_raw = blk: {
                var combined: [32]u8 = undefined;
                combined[0..16].* = (qs + j * 32)[0..16].*;
                combined[16..32].* = (qs + j * 32 + 16)[0..16].*;
                break :blk @as(@Vector(32, u8), combined);
            };
            const low_u8: @Vector(32, u8) = full_nibbles_raw & @as(@Vector(32, u8), @splat(0x0F));
            const high_u8: @Vector(32, u8) = full_nibbles_raw >> @as(@Vector(32, u8), @splat(4));

            // Sub-block 0: low nibbles
            const x_block_0 = sb * 8 + j * 2;
            {
                const input0: V32i8 = (x_vals_0 + x_block_0 * 32)[0..32].*;
                const input1: V32i8 = (x_vals_1 + x_block_0 * 32)[0..32].*;

                const dot0_sum: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, low_u8, input0)));
                const sum0_total: f32 = @floatFromInt(@reduce(.Add, intDotI8(zero_i32, input0, ones_i8)));
                sum0 += x_scales_0[x_block_0] * (sc1 * dot0_sum - m1 * sum0_total);

                const dot1_sum: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, low_u8, input1)));
                const sum1_total: f32 = @floatFromInt(@reduce(.Add, intDotI8(zero_i32, input1, ones_i8)));
                sum1 += x_scales_1[x_block_0] * (sc1 * dot1_sum - m1 * sum1_total);
            }

            // Sub-block 1: high nibbles
            const x_block_1 = sb * 8 + j * 2 + 1;
            {
                const input0: V32i8 = (x_vals_0 + x_block_1 * 32)[0..32].*;
                const input1: V32i8 = (x_vals_1 + x_block_1 * 32)[0..32].*;

                const dot0_sum: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, high_u8, input0)));
                const sum0_total: f32 = @floatFromInt(@reduce(.Add, intDotI8(zero_i32, input0, ones_i8)));
                sum0 += x_scales_0[x_block_1] * (sc2 * dot0_sum - m2 * sum0_total);

                const dot1_sum: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, high_u8, input1)));
                const sum1_total: f32 = @floatFromInt(@reduce(.Add, intDotI8(zero_i32, input1, ones_i8)));
                sum1 += x_scales_1[x_block_1] * (sc2 * dot1_sum - m2 * sum1_total);
            }
        }
    }

    return .{ sum0, sum1 };
}

// ---- Threaded matmul with Q8 input (for dense variant) ----

/// Matmul: quantizes f32 inputs to Q8, then dispatches Q4_K integer dot product.
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
    matmulQ8(pool, q_vals, q_scales, w_bytes, output, in_dim, out_dim, batch_size);
}

/// Matmul with pre-quantized i8 input, parallelized over output rows.
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
    const num_q8_blocks = in_dim / 32;
    const num_super_blocks = in_dim / Q4K_BLOCK_SIZE;
    const row_bytes = num_super_blocks * Q4K_BLOCK_BYTES;
    const num_tiles = out_dim;

    const Kernel = struct {
        fn run(
            quantized_vals: []const i8,
            quantized_scales: []const f32,
            out: []f32,
            weights: []const u8,
            input_dim: usize,
            output_dim: usize,
            batch: usize,
            n_q8_blocks: usize,
            n_super_blocks: usize,
            r_bytes: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            _ = input_dim;
            for (start_row..end_row) |row| {
                const weight_row = weights.ptr + row * r_bytes;
                var token: usize = 0;
                while (token + 1 < batch) : (token += 2) {
                    const results = dotQ4_K_q8_x2(
                        quantized_vals[token * n_q8_blocks * 32 ..].ptr,
                        quantized_scales[token * n_q8_blocks ..].ptr,
                        quantized_vals[(token + 1) * n_q8_blocks * 32 ..].ptr,
                        quantized_scales[(token + 1) * n_q8_blocks ..].ptr,
                        weight_row,
                        n_super_blocks,
                    );
                    out[token * output_dim + row] = results[0];
                    out[(token + 1) * output_dim + row] = results[1];
                }
                if (token < batch) {
                    out[token * output_dim + row] = dotQ4_K_q8(
                        quantized_vals[token * n_q8_blocks * 32 ..].ptr,
                        quantized_scales[token * n_q8_blocks ..].ptr,
                        weight_row,
                        n_super_blocks,
                    );
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
                    p.spawnWg(&wg, Kernel.run, .{
                        q_vals, q_scales, output, w_bytes,
                        in_dim, out_dim, batch_size,
                        num_q8_blocks, num_super_blocks, row_bytes,
                        start, start + count,
                    });
                    start += count;
                }
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(
        q_vals, q_scales, output, w_bytes,
        in_dim, out_dim, batch_size,
        num_q8_blocks, num_super_blocks, row_bytes,
        0, num_tiles,
    );
}

/// Fused gate+up projections with SiLU*hadamard, parallelized.
/// Quantizes f32 inputs, then dispatches Q4_K integer dot product.
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
    matmulSiluHadamardQ8(pool, q_vals, q_scales, gate_weights, up_weights, output, in_dim, out_dim, batch_size);
}

/// Fused gate+up with SiLU*hadamard using pre-quantized i8 input.
pub fn matmulSiluHadamardQ8(
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
    const num_q8_blocks = in_dim / 32;
    const num_super_blocks = in_dim / Q4K_BLOCK_SIZE;
    const row_bytes = num_super_blocks * Q4K_BLOCK_BYTES;
    const num_tiles = out_dim;

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
            n_q8_blocks: usize,
            n_super_blocks: usize,
            r_bytes: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            _ = input_dim;
            for (start_row..end_row) |row| {
                const gate_row = gate_w.ptr + row * r_bytes;
                const up_row = up_w.ptr + row * r_bytes;
                var token: usize = 0;
                while (token + 1 < batch) : (token += 2) {
                    const v0 = quantized_vals[token * n_q8_blocks * 32 ..].ptr;
                    const s0 = quantized_scales[token * n_q8_blocks ..].ptr;
                    const v1 = quantized_vals[(token + 1) * n_q8_blocks * 32 ..].ptr;
                    const s1 = quantized_scales[(token + 1) * n_q8_blocks ..].ptr;
                    const gate_results = dotQ4_K_q8_x2(v0, s0, v1, s1, gate_row, n_super_blocks);
                    const up_results = dotQ4_K_q8_x2(v0, s0, v1, s1, up_row, n_super_blocks);
                    inline for (0..2) |t| {
                        const gate = gate_results[t];
                        out[(token + t) * output_dim + row] = common.silu(gate) * up_results[t];
                    }
                }
                if (token < batch) {
                    const v = quantized_vals[token * n_q8_blocks * 32 ..].ptr;
                    const s = quantized_scales[token * n_q8_blocks ..].ptr;
                    const gate = dotQ4_K_q8(v, s, gate_row, n_super_blocks);
                    const up = dotQ4_K_q8(v, s, up_row, n_super_blocks);
                    out[token * output_dim + row] = common.silu(gate) * up;
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
                    p.spawnWg(&wg, Kernel.run, .{
                        q_vals, q_scales, output, gate_weights, up_weights,
                        in_dim, out_dim, batch_size,
                        num_q8_blocks, num_super_blocks, row_bytes,
                        start, start + count,
                    });
                    start += count;
                }
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(
        q_vals, q_scales, output, gate_weights, up_weights,
        in_dim, out_dim, batch_size,
        num_q8_blocks, num_super_blocks, row_bytes,
        0, num_tiles,
    );
}

// ---- Scalar matmul with f32 input (for MOE variant) ----

/// Matmul with Q4_K quantized weights and f32 input, parallelized over output rows.
/// No Q8 input quantization — dequantizes Q4_K inline. Used by MOE variants.
pub fn matmulScalar(pool: ?*Pool, input: []const f32, w_bytes: []const u8, output: []f32, in_dim: usize, out_dim: usize, batch_size: usize) void {
    const Kernel = struct {
        fn run(
            batch_input: []const f32,
            batch_output: []f32,
            weights: []const u8,
            input_dim: usize,
            output_dim: usize,
            batch: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            @setFloatMode(.optimized);
            const num_super_blocks = input_dim / Q4K_BLOCK_SIZE;
            const row_bytes = num_super_blocks * Q4K_BLOCK_BYTES;

            for (start_row..end_row) |row| {
                for (0..batch) |token| {
                    const input_slice = batch_input[token * input_dim ..][0..input_dim];
                    var sum: f32 = 0.0;

                    for (0..num_super_blocks) |sb| {
                        const block = weights[row * row_bytes + sb * Q4K_BLOCK_BYTES ..][0..Q4K_BLOCK_BYTES];
                        const d_val: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                        const dmin_val: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[2..4], .little))));
                        const sm = unpackScales(block[4..16]);
                        const qs = block[Q4K_HEADER_BYTES..Q4K_BLOCK_BYTES];

                        for (0..4) |j| {
                            const sc1: f32 = d_val * @as(f32, @floatFromInt(sm.scales[j * 2]));
                            const m1: f32 = dmin_val * @as(f32, @floatFromInt(sm.mins[j * 2]));
                            const sc2: f32 = d_val * @as(f32, @floatFromInt(sm.scales[j * 2 + 1]));
                            const m2: f32 = dmin_val * @as(f32, @floatFromInt(sm.mins[j * 2 + 1]));

                            const base_idx = sb * Q4K_BLOCK_SIZE + j * 64;
                            for (0..32) |l| {
                                const byte = qs[j * 32 + l];
                                sum += input_slice[base_idx + l] * (sc1 * @as(f32, @floatFromInt(byte & 0x0F)) - m1);
                                sum += input_slice[base_idx + 32 + l] * (sc2 * @as(f32, @floatFromInt(byte >> 4)) - m2);
                            }
                        }
                    }

                    batch_output[token * output_dim + row] = sum;
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
                p.spawnWg(&wg, Kernel.run, .{ input, output, w_bytes, in_dim, out_dim, batch_size, start, start + count });
                start += count;
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(input, output, w_bytes, in_dim, out_dim, batch_size, 0, out_dim);
}

/// Fused gate+up projections with SiLU*hadamard, f32 input, parallelized.
/// No Q8 input quantization — dequantizes Q4_K inline. Used by MOE variants.
pub fn matmulSiluHadamardScalar(pool: ?*Pool, input: []const f32, gate_w: []const u8, up_w: []const u8, output: []f32, in_dim: usize, out_dim: usize, batch_size: usize) void {
    const Kernel = struct {
        fn run(
            batch_input: []const f32,
            batch_output: []f32,
            gate_weights: []const u8,
            up_weights: []const u8,
            input_dim: usize,
            output_dim: usize,
            batch: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            @setFloatMode(.optimized);
            const num_super_blocks = input_dim / Q4K_BLOCK_SIZE;
            const row_bytes = num_super_blocks * Q4K_BLOCK_BYTES;

            for (start_row..end_row) |row| {
                for (0..batch) |token| {
                    const input_slice = batch_input[token * input_dim ..][0..input_dim];
                    var gate_sum: f32 = 0.0;
                    var up_sum: f32 = 0.0;

                    for (0..num_super_blocks) |sb| {
                        const gate_block = gate_weights[row * row_bytes + sb * Q4K_BLOCK_BYTES ..][0..Q4K_BLOCK_BYTES];
                        const up_block = up_weights[row * row_bytes + sb * Q4K_BLOCK_BYTES ..][0..Q4K_BLOCK_BYTES];

                        const gate_d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, gate_block[0..2], .little))));
                        const gate_dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, gate_block[2..4], .little))));
                        const gate_sm = unpackScales(gate_block[4..16]);
                        const gate_qs = gate_block[Q4K_HEADER_BYTES..Q4K_BLOCK_BYTES];

                        const up_d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, up_block[0..2], .little))));
                        const up_dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, up_block[2..4], .little))));
                        const up_sm = unpackScales(up_block[4..16]);
                        const up_qs = up_block[Q4K_HEADER_BYTES..Q4K_BLOCK_BYTES];

                        for (0..4) |j| {
                            const gate_sc1: f32 = gate_d * @as(f32, @floatFromInt(gate_sm.scales[j * 2]));
                            const gate_m1: f32 = gate_dmin * @as(f32, @floatFromInt(gate_sm.mins[j * 2]));
                            const gate_sc2: f32 = gate_d * @as(f32, @floatFromInt(gate_sm.scales[j * 2 + 1]));
                            const gate_m2: f32 = gate_dmin * @as(f32, @floatFromInt(gate_sm.mins[j * 2 + 1]));
                            const up_sc1: f32 = up_d * @as(f32, @floatFromInt(up_sm.scales[j * 2]));
                            const up_m1: f32 = up_dmin * @as(f32, @floatFromInt(up_sm.mins[j * 2]));
                            const up_sc2: f32 = up_d * @as(f32, @floatFromInt(up_sm.scales[j * 2 + 1]));
                            const up_m2: f32 = up_dmin * @as(f32, @floatFromInt(up_sm.mins[j * 2 + 1]));

                            const base_idx = sb * Q4K_BLOCK_SIZE + j * 64;
                            for (0..32) |l| {
                                const gate_byte = gate_qs[j * 32 + l];
                                const up_byte = up_qs[j * 32 + l];
                                const inp_lo = input_slice[base_idx + l];
                                const inp_hi = input_slice[base_idx + 32 + l];
                                gate_sum += inp_lo * (gate_sc1 * @as(f32, @floatFromInt(gate_byte & 0x0F)) - gate_m1);
                                gate_sum += inp_hi * (gate_sc2 * @as(f32, @floatFromInt(gate_byte >> 4)) - gate_m2);
                                up_sum += inp_lo * (up_sc1 * @as(f32, @floatFromInt(up_byte & 0x0F)) - up_m1);
                                up_sum += inp_hi * (up_sc2 * @as(f32, @floatFromInt(up_byte >> 4)) - up_m2);
                            }
                        }
                    }

                    batch_output[token * output_dim + row] = common.silu(gate_sum) * up_sum;
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
                p.spawnWg(&wg, Kernel.run, .{ input, output, gate_w, up_w, in_dim, out_dim, batch_size, start, start + count });
                start += count;
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(input, output, gate_w, up_w, in_dim, out_dim, batch_size, 0, out_dim);
}

// ---- SIMD primitives ----

/// Unsigned u8 × signed i8 dot product with i32 accumulation.
/// result[i] = acc[i] + sum(u[4i+j] * s[4i+j]) for j=0..3.
/// On x86 this maps directly to vpdpbusd (unsigned × signed).
inline fn intDotU8xI8(
    acc: @Vector(8, i32),
    unsigned: @Vector(32, u8),
    signed: @Vector(32, i8),
) @Vector(8, i32) {
    if (builtin.mode == .Debug) {
        var dots: [8]i32 = acc;
        inline for (0..8) |i| {
            inline for (0..4) |j| {
                const idx = i * 4 + j;
                dots[i] += @as(i32, @intCast(unsigned[idx])) * @as(i32, signed[idx]);
            }
        }
        return dots;
    }

    if (comptime builtin.cpu.arch == .x86_64) {
        // vpdpbusd: unsigned byte × signed byte with i32 accumulation.
        // The sign trick is NOT needed here because nibbles are already unsigned (0-15).
        // However, the signed input may be negative. vpdpbusd treats first operand as
        // unsigned and second as signed, which is exactly what we have (unsigned nibbles,
        // signed Q8 values). But the operands are swapped: we need u8 * i8.
        //
        // vpdpbusd(acc, A_unsigned, B_signed) = acc + sum(A_unsigned[4i+j] * B_signed[4i+j])
        // We have: unsigned nibbles (0-15) and signed Q8 input (-127..127).
        // Pass nibbles as first (unsigned) operand, Q8 as second (signed) operand.
        return asm ("{vex} vpdpbusd %[b], %[a], %[out]"
            : [out] "=x" (-> @Vector(8, i32)),
            : [a] "x" (unsigned),
              [b] "x" (@as(@Vector(32, u8), @bitCast(signed))),
              [tied] "0" (acc),
        );
    }

    if (comptime builtin.cpu.arch == .aarch64) {
        if (comptime std.Target.aarch64.featureSetHas(builtin.cpu.features, .dotprod)) {
            // NEON: use sign trick to convert to signed×signed for sdot.
            // abs(signed) is safe since Q8 values are in [-127, 127] (never -128).
            const zero: @Vector(32, i8) = @splat(0);
            const s = signed;
            const abs_s = @abs(s);
            const s_negative = s < zero;
            const u_as_i8: @Vector(32, i8) = @bitCast(unsigned);
            const corrected_u: @Vector(32, i8) = @select(i8, s_negative, zero -% u_as_i8, u_as_i8);

            const u_arr: [32]u8 = @bitCast(abs_s);
            const s_arr: [32]i8 = corrected_u;
            const a: [8]i32 = acc;

            const res_lo = asm ("sdot %[out].4s, %[a].16b, %[b].16b"
                : [out] "=w" (-> @Vector(4, i32)),
                : [a] "w" (@as(@Vector(16, i8), @bitCast(u_arr[0..16].*))),
                  [b] "w" (@as(@Vector(16, i8), s_arr[0..16].*)),
                  [tied] "0" (@as(@Vector(4, i32), a[0..4].*)),
            );
            const res_hi = asm ("sdot %[out].4s, %[a].16b, %[b].16b"
                : [out] "=w" (-> @Vector(4, i32)),
                : [a] "w" (@as(@Vector(16, i8), @bitCast(u_arr[16..32].*))),
                  [b] "w" (@as(@Vector(16, i8), s_arr[16..32].*)),
                  [tied] "0" (@as(@Vector(4, i32), a[4..8].*)),
            );

            var result: [8]i32 = undefined;
            result[0..4].* = res_lo;
            result[4..8].* = res_hi;
            return result;
        }
    }

    // Scalar fallback.
    var dots: [8]i32 = acc;
    inline for (0..8) |i| {
        inline for (0..4) |j| {
            const idx = i * 4 + j;
            dots[i] += @as(i32, @intCast(unsigned[idx])) * @as(i32, signed[idx]);
        }
    }
    return dots;
}

/// Signed i8 × signed i8 dot product with i32 accumulation (from q4_0).
/// Used here for computing sum(x_q[i]) via dot(x_q, ones).
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

    // x86 / fallback: apply sign trick, then delegate to vpdpbusd.
    const zero: @Vector(32, i8) = @splat(0);
    const abs_a = @abs(a);
    const a_negative = a < zero;
    const signed_b: @Vector(32, u8) = @bitCast(@select(i8, a_negative, zero -% b, b));
    return vpdpbusd(acc, abs_a, signed_b);
}

/// Unsigned x signed byte dot product with i32 accumulation (vpdpbusd).
inline fn vpdpbusd(
    acc: @Vector(8, i32),
    unsigned: @Vector(32, u8),
    signed: @Vector(32, u8),
) @Vector(8, i32) {
    if (builtin.mode == .Debug) {
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
