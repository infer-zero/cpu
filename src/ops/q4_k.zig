const std = @import("std");
const builtin = @import("builtin");
const common = @import("common.zig");
const thread_pool = @import("../thread_pool.zig");

const Pool = thread_pool.Pool;
const WaitGroup = thread_pool.WaitGroup;

/// Q4_K super-block: 256 elements in 144 bytes.
/// Layout: d[2](f16) + dmin[2](f16) + scales[12](6-bit packed) + qs[128](nibbles).
const Q4K_BLOCK_SIZE: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;
const Q4K_HEADER_BYTES: usize = 16; // d + dmin + scales
const Q4K_DATA_BYTES: usize = 128; // qs

/// Tokens dotted per weight-block decode in the Q8_K R4 microkernel. Each
/// (super-block, row) weight unpack is reused across this many tokens; larger
/// amortizes the unpack/L1 weight read but needs `TILE` i32 accumulators per
/// row in the AVX2 register file (16 ymm). 4 is the throughput sweet spot.
const Q8K_TOKEN_TILE: usize = 4;

/// True when targeting a CPU with 512-bit VNNI (AVX-512 VNNI).
const has_avx512_vnni = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vnni);

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

/// Precompute Q8 block sums for bias correction.
/// Each Q8 block sum is sum(x_q[i]) for i in 0..31, needed for the dmin*min term.
/// Computed once per token, reused across all output rows.
fn precomputeQ8BlockSums(
    x_vals: [*]const i8,
    num_q8_blocks: usize,
    out_sums: [*]f32,
) void {
    const V32i8 = @Vector(32, i8);
    const zero_i32: @Vector(8, i32) = @splat(0);
    const ones_i8: V32i8 = @splat(1);
    for (0..num_q8_blocks) |block| {
        const input: V32i8 = (x_vals + block * 32)[0..32].*;
        out_sums[block] = @floatFromInt(@reduce(.Add, intDotI8(zero_i32, input, ones_i8)));
    }
}

/// Dot product of Q4_K weight row × Q8-quantized input vector.
/// Each Q4_K super-block (256 elements) consumes 8 consecutive Q8 blocks (32 elements each).
/// `x_block_sums` contains precomputed sum(x_q[i]) per Q8 block, avoiding redundant
/// SIMD reductions across output rows.
///
/// Math per sub-block of 32 elements:
///   partial = x_scale * (d * scale_j * dot(x_q, nibble) - dmin * min_j * sum(x_q))
inline fn dotQ4_K_q8(
    noalias x_vals: [*]const i8,
    noalias x_scales: [*]const f32,
    noalias x_block_sums: [*]const f32,
    noalias w_bytes: [*]const u8,
    num_super_blocks: usize,
) f32 {
    @setFloatMode(.optimized);
    const zero_i32: @Vector(8, i32) = @splat(0);

    var acc: f32 = 0;

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

            const full_nibbles_raw: @Vector(32, u8) = (qs + j * 32)[0..32].*;
            const low_u8: @Vector(32, u8) = full_nibbles_raw & @as(@Vector(32, u8), @splat(0x0F));
            const high_u8: @Vector(32, u8) = full_nibbles_raw >> @as(@Vector(32, u8), @splat(4));

            const x_block_0 = sb * 8 + j * 2;
            const input_lo: @Vector(32, i8) = (x_vals + x_block_0 * 32)[0..32].*;
            const dot_lo: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, low_u8, input_lo)));
            acc += x_scales[x_block_0] * (sc1 * dot_lo - m1 * x_block_sums[x_block_0]);

            const x_block_1 = sb * 8 + j * 2 + 1;
            const input_hi: @Vector(32, i8) = (x_vals + x_block_1 * 32)[0..32].*;
            const dot_hi: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, high_u8, input_hi)));
            acc += x_scales[x_block_1] * (sc2 * dot_hi - m2 * x_block_sums[x_block_1]);
        }
    }

    return acc;
}

/// 2-token variant: unpack Q4_K weight block once, compute dot products for 2 tokens.
inline fn dotQ4_K_q8_x2(
    noalias x_vals_0: [*]const i8,
    noalias x_scales_0: [*]const f32,
    noalias x_block_sums_0: [*]const f32,
    noalias x_vals_1: [*]const i8,
    noalias x_scales_1: [*]const f32,
    noalias x_block_sums_1: [*]const f32,
    noalias w_bytes: [*]const u8,
    num_super_blocks: usize,
) [2]f32 {
    @setFloatMode(.optimized);
    const zero_i32: @Vector(8, i32) = @splat(0);

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

            const full_nibbles_raw: @Vector(32, u8) = (qs + j * 32)[0..32].*;
            const low_u8: @Vector(32, u8) = full_nibbles_raw & @as(@Vector(32, u8), @splat(0x0F));
            const high_u8: @Vector(32, u8) = full_nibbles_raw >> @as(@Vector(32, u8), @splat(4));

            const x_block_0 = sb * 8 + j * 2;
            {
                const input0: @Vector(32, i8) = (x_vals_0 + x_block_0 * 32)[0..32].*;
                const input1: @Vector(32, i8) = (x_vals_1 + x_block_0 * 32)[0..32].*;
                const dot0: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, low_u8, input0)));
                const dot1: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, low_u8, input1)));
                sum0 += x_scales_0[x_block_0] * (sc1 * dot0 - m1 * x_block_sums_0[x_block_0]);
                sum1 += x_scales_1[x_block_0] * (sc1 * dot1 - m1 * x_block_sums_1[x_block_0]);
            }

            const x_block_1 = sb * 8 + j * 2 + 1;
            {
                const input0: @Vector(32, i8) = (x_vals_0 + x_block_1 * 32)[0..32].*;
                const input1: @Vector(32, i8) = (x_vals_1 + x_block_1 * 32)[0..32].*;
                const dot0: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, high_u8, input0)));
                const dot1: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, high_u8, input1)));
                sum0 += x_scales_0[x_block_1] * (sc2 * dot0 - m2 * x_block_sums_0[x_block_1]);
                sum1 += x_scales_1[x_block_1] * (sc2 * dot1 - m2 * x_block_sums_1[x_block_1]);
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
/// Precomputes Q8 block sums once, then reuses across all output rows.
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

    // Heap-allocate (not stack) — worst case at batch=4096, dim=32768 is
    // 16 MB, which overflows the default 8 MB thread stack. Sizing to
    // actual need (batch_size * num_q8_blocks) keeps typical decode calls
    // in a single page.
    const block_sums_buf = std.heap.page_allocator.alloc(f32, batch_size * num_q8_blocks) catch @panic("OOM allocating block_sums_buf");
    defer std.heap.page_allocator.free(block_sums_buf);
    for (0..batch_size) |token| {
        precomputeQ8BlockSums(
            q_vals[token * num_q8_blocks * 32 ..].ptr,
            num_q8_blocks,
            block_sums_buf[token * num_q8_blocks ..].ptr,
        );
    }
    const block_sums: []const f32 = block_sums_buf;

    const Kernel = struct {
        fn run(
            quantized_vals: []const i8,
            quantized_scales: []const f32,
            q8_block_sums: []const f32,
            out: []f32,
            weights: []const u8,
            output_dim: usize,
            batch: usize,
            n_q8_blocks: usize,
            n_super_blocks: usize,
            r_bytes: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            for (start_row..end_row) |row| {
                const weight_row = weights.ptr + row * r_bytes;
                var token: usize = 0;
                while (token + 1 < batch) : (token += 2) {
                    const results = dotQ4_K_q8_x2(
                        quantized_vals[token * n_q8_blocks * 32 ..].ptr,
                        quantized_scales[token * n_q8_blocks ..].ptr,
                        q8_block_sums[token * n_q8_blocks ..].ptr,
                        quantized_vals[(token + 1) * n_q8_blocks * 32 ..].ptr,
                        quantized_scales[(token + 1) * n_q8_blocks ..].ptr,
                        q8_block_sums[(token + 1) * n_q8_blocks ..].ptr,
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
                        q8_block_sums[token * n_q8_blocks ..].ptr,
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
                        q_vals, q_scales, block_sums, output, w_bytes,
                        out_dim, batch_size,
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
        q_vals, q_scales, block_sums, output, w_bytes,
        out_dim, batch_size,
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
/// Precomputes Q8 block sums once, then reuses across all output rows.
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

    // Heap-allocate (not stack) — see matmulQ8 for rationale.
    const block_sums_buf = std.heap.page_allocator.alloc(f32, batch_size * num_q8_blocks) catch @panic("OOM allocating block_sums_buf");
    defer std.heap.page_allocator.free(block_sums_buf);
    for (0..batch_size) |token| {
        precomputeQ8BlockSums(
            q_vals[token * num_q8_blocks * 32 ..].ptr,
            num_q8_blocks,
            block_sums_buf[token * num_q8_blocks ..].ptr,
        );
    }
    const block_sums: []const f32 = block_sums_buf;

    const Kernel = struct {
        fn run(
            quantized_vals: []const i8,
            quantized_scales: []const f32,
            q8_block_sums: []const f32,
            out: []f32,
            gate_w: []const u8,
            up_w: []const u8,
            output_dim: usize,
            batch: usize,
            n_q8_blocks: usize,
            n_super_blocks: usize,
            r_bytes: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            for (start_row..end_row) |row| {
                const gate_row = gate_w.ptr + row * r_bytes;
                const up_row = up_w.ptr + row * r_bytes;
                var token: usize = 0;
                while (token + 1 < batch) : (token += 2) {
                    const v0 = quantized_vals[token * n_q8_blocks * 32 ..].ptr;
                    const s0 = quantized_scales[token * n_q8_blocks ..].ptr;
                    const bs0 = q8_block_sums[token * n_q8_blocks ..].ptr;
                    const v1 = quantized_vals[(token + 1) * n_q8_blocks * 32 ..].ptr;
                    const s1 = quantized_scales[(token + 1) * n_q8_blocks ..].ptr;
                    const bs1 = q8_block_sums[(token + 1) * n_q8_blocks ..].ptr;
                    const gate_results = dotQ4_K_q8_x2(v0, s0, bs0, v1, s1, bs1, gate_row, n_super_blocks);
                    const up_results = dotQ4_K_q8_x2(v0, s0, bs0, v1, s1, bs1, up_row, n_super_blocks);
                    inline for (0..2) |t| {
                        const gate = gate_results[t];
                        out[(token + t) * output_dim + row] = common.silu(gate) * up_results[t];
                    }
                }
                if (token < batch) {
                    const v = quantized_vals[token * n_q8_blocks * 32 ..].ptr;
                    const s = quantized_scales[token * n_q8_blocks ..].ptr;
                    const bs = q8_block_sums[token * n_q8_blocks ..].ptr;
                    const gate = dotQ4_K_q8(v, s, bs, gate_row, n_super_blocks);
                    const up = dotQ4_K_q8(v, s, bs, up_row, n_super_blocks);
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
                        q_vals, q_scales, block_sums, output, gate_weights, up_weights,
                        out_dim, batch_size,
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
        q_vals, q_scales, block_sums, output, gate_weights, up_weights,
        out_dim, batch_size,
        num_q8_blocks, num_super_blocks, row_bytes,
        0, num_tiles,
    );
}

// ---- R4 matmul (4-row interleaved Q4_K) ----

/// Repack Q4_K weights from row-major to R4 (4-row interleaved) layout.
/// Groups of 4 rows have their super-blocks interleaved at block granularity.
/// Caller must free the returned slice. Frees the input `raw` slice.
pub fn repackQ4KR4(allocator: std.mem.Allocator, raw: []const u8, in_dim: usize, out_dim: usize) ![]const u8 {
    const num_super_blocks = in_dim / Q4K_BLOCK_SIZE;
    const row_bytes = num_super_blocks * Q4K_BLOCK_BYTES;
    const padded_out_dim = (out_dim + 3) / 4 * 4;
    const num_groups = padded_out_dim / 4;

    const result = try allocator.alloc(u8, padded_out_dim * row_bytes);
    @memset(result, 0);

    for (0..num_groups) |group| {
        for (0..num_super_blocks) |sb_pos| {
            const dst_offset = (group * num_super_blocks + sb_pos) * (4 * Q4K_BLOCK_BYTES);
            for (0..4) |row_in_group| {
                const row = group * 4 + row_in_group;
                const dst = result[dst_offset + row_in_group * Q4K_BLOCK_BYTES ..][0..Q4K_BLOCK_BYTES];
                if (row < out_dim) {
                    const src_offset = row * row_bytes + sb_pos * Q4K_BLOCK_BYTES;
                    @memcpy(dst, raw[src_offset..][0..Q4K_BLOCK_BYTES]);
                }
            }
        }
    }

    allocator.free(@constCast(raw));
    return result;
}

/// Matmul with R4-interleaved Q4_K weights: quantizes f32 input, then dispatches R4 kernel.
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
    matmulQ8R4(pool, q_vals, q_scales, w_bytes, output, in_dim, out_dim, batch_size);
}

/// Matmul with pre-quantized i8 input and R4-interleaved Q4_K weights.
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
    const num_q8_blocks = in_dim / 32;
    const num_super_blocks = in_dim / Q4K_BLOCK_SIZE;
    const num_groups = (out_dim + 3) / 4;
    const num_tiles = num_groups;

    // Heap-allocate (not stack) — see matmulQ8 for rationale.
    const block_sums_buf = std.heap.page_allocator.alloc(f32, batch_size * num_q8_blocks) catch @panic("OOM allocating block_sums_buf");
    defer std.heap.page_allocator.free(block_sums_buf);
    for (0..batch_size) |token| {
        precomputeQ8BlockSums(
            q_vals[token * num_q8_blocks * 32 ..].ptr,
            num_q8_blocks,
            block_sums_buf[token * num_q8_blocks ..].ptr,
        );
    }
    const block_sums: []const f32 = block_sums_buf;

    const Kernel = struct {
        fn run(
            quantized_vals: []const i8,
            quantized_scales: []const f32,
            q8_block_sums: []const f32,
            out: []f32,
            weights: []const u8,
            output_dim: usize,
            batch: usize,
            n_q8_blocks: usize,
            n_super_blocks: usize,
            start_tile: usize,
            end_tile: usize,
            total_groups: usize,
        ) void {
            _ = total_groups;
            for (start_tile..end_tile) |group| {
                const base_row = group * 4;
                const group_ptr = weights.ptr + group * n_super_blocks * (4 * Q4K_BLOCK_BYTES);

                var token: usize = 0;
                while (token + 1 < batch) : (token += 2) {
                    const results = dotQ4_K_q8_R4x2(
                        quantized_vals[token * n_q8_blocks * 32 ..].ptr,
                        quantized_scales[token * n_q8_blocks ..].ptr,
                        q8_block_sums[token * n_q8_blocks ..].ptr,
                        quantized_vals[(token + 1) * n_q8_blocks * 32 ..].ptr,
                        quantized_scales[(token + 1) * n_q8_blocks ..].ptr,
                        q8_block_sums[(token + 1) * n_q8_blocks ..].ptr,
                        group_ptr,
                        n_super_blocks,
                    );
                    inline for (0..4) |r| {
                        if (base_row + r < output_dim) {
                            out[token * output_dim + base_row + r] = results[0][r];
                            out[(token + 1) * output_dim + base_row + r] = results[1][r];
                        }
                    }
                }
                if (token < batch) {
                    const results = dotQ4_K_q8_R4(
                        quantized_vals[token * n_q8_blocks * 32 ..].ptr,
                        quantized_scales[token * n_q8_blocks ..].ptr,
                        q8_block_sums[token * n_q8_blocks ..].ptr,
                        group_ptr,
                        n_super_blocks,
                    );
                    inline for (0..4) |r| {
                        if (base_row + r < output_dim) {
                            out[token * output_dim + base_row + r] = results[r];
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
                    p.spawnWg(&wg, Kernel.run, .{
                        q_vals, q_scales, block_sums, output, w_bytes,
                        out_dim, batch_size,
                        num_q8_blocks, num_super_blocks,
                        start, start + count, num_groups,
                    });
                    start += count;
                }
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(
        q_vals, q_scales, block_sums, output, w_bytes,
        out_dim, batch_size,
        num_q8_blocks, num_super_blocks,
        0, num_tiles, num_groups,
    );
}

/// Fused gate+up with SiLU*hadamard using R4-interleaved Q4_K weights.
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
    matmulSiluHadamardQ8R4(pool, q_vals, q_scales, gate_weights, up_weights, output, in_dim, out_dim, batch_size);
}

/// Fused gate+up with SiLU*hadamard using pre-quantized i8 input and R4-interleaved Q4_K weights.
pub fn matmulSiluHadamardQ8R4(
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
    const num_groups = (out_dim + 3) / 4;
    const num_tiles = num_groups;

    // Heap-allocate (not stack) — see matmulQ8 for rationale.
    const block_sums_buf = std.heap.page_allocator.alloc(f32, batch_size * num_q8_blocks) catch @panic("OOM allocating block_sums_buf");
    defer std.heap.page_allocator.free(block_sums_buf);
    for (0..batch_size) |token| {
        precomputeQ8BlockSums(
            q_vals[token * num_q8_blocks * 32 ..].ptr,
            num_q8_blocks,
            block_sums_buf[token * num_q8_blocks ..].ptr,
        );
    }
    const block_sums: []const f32 = block_sums_buf;

    const Kernel = struct {
        fn run(
            quantized_vals: []const i8,
            quantized_scales: []const f32,
            q8_block_sums: []const f32,
            out: []f32,
            gate_w: []const u8,
            up_w: []const u8,
            output_dim: usize,
            batch: usize,
            n_q8_blocks: usize,
            n_super_blocks: usize,
            start_tile: usize,
            end_tile: usize,
        ) void {
            for (start_tile..end_tile) |group| {
                const base_row = group * 4;
                const gate_group_ptr = gate_w.ptr + group * n_super_blocks * (4 * Q4K_BLOCK_BYTES);
                const up_group_ptr = up_w.ptr + group * n_super_blocks * (4 * Q4K_BLOCK_BYTES);

                var token: usize = 0;
                while (token + 1 < batch) : (token += 2) {
                    const v0 = quantized_vals[token * n_q8_blocks * 32 ..].ptr;
                    const s0 = quantized_scales[token * n_q8_blocks ..].ptr;
                    const bs0 = q8_block_sums[token * n_q8_blocks ..].ptr;
                    const v1 = quantized_vals[(token + 1) * n_q8_blocks * 32 ..].ptr;
                    const s1 = quantized_scales[(token + 1) * n_q8_blocks ..].ptr;
                    const bs1 = q8_block_sums[(token + 1) * n_q8_blocks ..].ptr;
                    const gate_results = dotQ4_K_q8_R4x2(v0, s0, bs0, v1, s1, bs1, gate_group_ptr, n_super_blocks);
                    const up_results = dotQ4_K_q8_R4x2(v0, s0, bs0, v1, s1, bs1, up_group_ptr, n_super_blocks);
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
                    const v = quantized_vals[token * n_q8_blocks * 32 ..].ptr;
                    const s = quantized_scales[token * n_q8_blocks ..].ptr;
                    const bs = q8_block_sums[token * n_q8_blocks ..].ptr;
                    const gate_results = dotQ4_K_q8_R4(v, s, bs, gate_group_ptr, n_super_blocks);
                    const up_results = dotQ4_K_q8_R4(v, s, bs, up_group_ptr, n_super_blocks);
                    inline for (0..4) |r| {
                        if (base_row + r < output_dim) {
                            out[token * output_dim + base_row + r] = common.silu(gate_results[r]) * up_results[r];
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
                    p.spawnWg(&wg, Kernel.run, .{
                        q_vals, q_scales, block_sums, output, gate_weights, up_weights,
                        out_dim, batch_size,
                        num_q8_blocks, num_super_blocks,
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
        q_vals, q_scales, block_sums, output, gate_weights, up_weights,
        out_dim, batch_size,
        num_q8_blocks, num_super_blocks,
        0, num_tiles,
    );
}

/// 4-row R4 dot product kernel for Q4_K weights.
/// Returns 4 dot products (one per row in the group).
/// Input Q8 blocks are loaded once and reused across all 4 weight rows.
inline fn dotQ4_K_q8_R4(
    noalias x_vals: [*]const i8,
    noalias x_scales: [*]const f32,
    noalias x_block_sums: [*]const f32,
    noalias group_ptr: [*]const u8,
    num_super_blocks: usize,
) [4]f32 {
    @setFloatMode(.optimized);
    @setEvalBranchQuota(50000);
    const zero_i32: @Vector(8, i32) = @splat(0);

    var acc: [4]f32 = .{ 0, 0, 0, 0 };

    for (0..num_super_blocks) |sb| {
        const x_base = sb * 8;

        inline for (0..4) |row| {
            const block_ptr = group_ptr + sb * (4 * Q4K_BLOCK_BYTES) + row * Q4K_BLOCK_BYTES;

            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
            const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[2..4], .little))));
            const sm = unpackScales(block_ptr[4..16]);
            const qs = block_ptr + Q4K_HEADER_BYTES;

            for (0..4) |j| {
                const sc1: f32 = d * @as(f32, @floatFromInt(sm.scales[j * 2]));
                const m1: f32 = dmin * @as(f32, @floatFromInt(sm.mins[j * 2]));
                const sc2: f32 = d * @as(f32, @floatFromInt(sm.scales[j * 2 + 1]));
                const m2: f32 = dmin * @as(f32, @floatFromInt(sm.mins[j * 2 + 1]));

                const full_nibbles_raw: @Vector(32, u8) = (qs + j * 32)[0..32].*;
                const low_u8: @Vector(32, u8) = full_nibbles_raw & @as(@Vector(32, u8), @splat(0x0F));
                const high_u8: @Vector(32, u8) = full_nibbles_raw >> @as(@Vector(32, u8), @splat(4));

                const x_block_0 = x_base + j * 2;
                const input_lo: @Vector(32, i8) = (x_vals + x_block_0 * 32)[0..32].*;
                const dot_lo: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, low_u8, input_lo)));
                acc[row] += x_scales[x_block_0] * (sc1 * dot_lo - m1 * x_block_sums[x_block_0]);

                const x_block_1 = x_base + j * 2 + 1;
                const input_hi: @Vector(32, i8) = (x_vals + x_block_1 * 32)[0..32].*;
                const dot_hi: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, high_u8, input_hi)));
                acc[row] += x_scales[x_block_1] * (sc2 * dot_hi - m2 * x_block_sums[x_block_1]);
            }
        }
    }

    return acc;
}

/// 2-token × 4-row R4 dot product kernel for Q4_K weights.
inline fn dotQ4_K_q8_R4x2(
    noalias x_vals_0: [*]const i8,
    noalias x_scales_0: [*]const f32,
    noalias x_block_sums_0: [*]const f32,
    noalias x_vals_1: [*]const i8,
    noalias x_scales_1: [*]const f32,
    noalias x_block_sums_1: [*]const f32,
    noalias group_ptr: [*]const u8,
    num_super_blocks: usize,
) [2][4]f32 {
    @setFloatMode(.optimized);
    @setEvalBranchQuota(50000);
    const zero_i32: @Vector(8, i32) = @splat(0);

    var acc0: [4]f32 = .{ 0, 0, 0, 0 };
    var acc1: [4]f32 = .{ 0, 0, 0, 0 };

    for (0..num_super_blocks) |sb| {
        const x_base = sb * 8;

        inline for (0..4) |row| {
            const block_ptr = group_ptr + sb * (4 * Q4K_BLOCK_BYTES) + row * Q4K_BLOCK_BYTES;

            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
            const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[2..4], .little))));
            const sm = unpackScales(block_ptr[4..16]);
            const qs = block_ptr + Q4K_HEADER_BYTES;

            for (0..4) |j| {
                const sc1: f32 = d * @as(f32, @floatFromInt(sm.scales[j * 2]));
                const m1: f32 = dmin * @as(f32, @floatFromInt(sm.mins[j * 2]));
                const sc2: f32 = d * @as(f32, @floatFromInt(sm.scales[j * 2 + 1]));
                const m2: f32 = dmin * @as(f32, @floatFromInt(sm.mins[j * 2 + 1]));

                const full_nibbles_raw: @Vector(32, u8) = (qs + j * 32)[0..32].*;
                const low_u8: @Vector(32, u8) = full_nibbles_raw & @as(@Vector(32, u8), @splat(0x0F));
                const high_u8: @Vector(32, u8) = full_nibbles_raw >> @as(@Vector(32, u8), @splat(4));

                const x_block_0 = x_base + j * 2;
                {
                    const in0: @Vector(32, i8) = (x_vals_0 + x_block_0 * 32)[0..32].*;
                    const in1: @Vector(32, i8) = (x_vals_1 + x_block_0 * 32)[0..32].*;
                    const d0: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, low_u8, in0)));
                    const d1: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, low_u8, in1)));
                    acc0[row] += x_scales_0[x_block_0] * (sc1 * d0 - m1 * x_block_sums_0[x_block_0]);
                    acc1[row] += x_scales_1[x_block_0] * (sc1 * d1 - m1 * x_block_sums_1[x_block_0]);
                }

                const x_block_1 = x_base + j * 2 + 1;
                {
                    const in0: @Vector(32, i8) = (x_vals_0 + x_block_1 * 32)[0..32].*;
                    const in1: @Vector(32, i8) = (x_vals_1 + x_block_1 * 32)[0..32].*;
                    const d0: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, high_u8, in0)));
                    const d1: f32 = @floatFromInt(@reduce(.Add, intDotU8xI8(zero_i32, high_u8, in1)));
                    acc0[row] += x_scales_0[x_block_1] * (sc2 * d0 - m2 * x_block_sums_0[x_block_1]);
                    acc1[row] += x_scales_1[x_block_1] * (sc2 * d1 - m2 * x_block_sums_1[x_block_1]);
                }
            }
        }
    }

    return .{ acc0, acc1 };
}

// ---- Q8_K activation path (wider integer-accumulating R4 microkernel) ----
//
// The R4 kernels above carry a per-32 f32 activation scale (Q8_0-style), so
// each 32-element sub-block must be reduced to f32 and scaled immediately --
// 8 horizontal reductions per super-block. This path instead quantizes the
// activation Q8_K-style: ONE f32 scale per 256-element super-block plus
// integer per-32 block sums. That lets the dot accumulate the whole
// super-block in i32 (the 6-bit weight scales applied in-SIMD) with a single
// f32 multiply per super-block -- one reduction instead of eight. Mirrors
// llama.cpp's `ggml_vec_dot_q4_K_q8_K`. Weights are unchanged (same R4 repack).

/// Q8_K-style activation quant: per-256 f32 scale + per-32 i32 block sums.
/// `q_d` is [in/256], `q_bsums` is [in/256 * 8] (sum of each 32-elem block).
pub fn quantizeF32ToQ8K(
    input: []const f32,
    q_vals: []i8,
    q_d: []f32,
    q_bsums: []i32,
) void {
    @setFloatMode(.optimized);
    const SB = Q4K_BLOCK_SIZE; // 256
    const V = @Vector(32, f32);
    const num_sb = input.len / SB;
    for (0..num_sb) |sb| {
        // Max-abs over the 256-element super-block, vectorized (auto-vec is
        // off on Zig 0.16, so this must be explicit @Vector).
        var amax_v: V = @splat(0);
        inline for (0..8) |g| {
            const chunk: V = input[sb * SB + g * 32 ..][0..32].*;
            amax_v = @max(amax_v, @abs(chunk));
        }
        const amax = @reduce(.Max, amax_v);
        if (amax == 0) {
            @memset(q_vals[sb * SB ..][0..SB], 0);
            q_d[sb] = 0;
            for (0..8) |g| q_bsums[sb * 8 + g] = 0;
            continue;
        }
        const inv: V = @splat(127.0 / amax);
        const lo: V = @splat(-127.0);
        const hi: V = @splat(127.0);
        q_d[sb] = amax / 127.0;
        inline for (0..8) |g| {
            const chunk: V = input[sb * SB + g * 32 ..][0..32].*;
            const clamped = @max(lo, @min(hi, @round(chunk * inv)));
            const qi: @Vector(32, i8) = @intFromFloat(clamped);
            q_vals[sb * SB + g * 32 ..][0..32].* = qi;
            q_bsums[sb * 8 + g] = @reduce(.Add, @as(@Vector(32, i32), qi));
        }
    }
}

/// 4-row R4 dot with Q8_K activation. Accumulates a whole super-block in i32
/// (one reduction at the end) per row. Returns 4 dot products.
inline fn dotQ4_K_q8K_R4(
    noalias x_vals: [*]const i8,
    noalias x_d: [*]const f32,
    noalias x_bsums: [*]const i32,
    noalias group_ptr: [*]const u8,
    num_super_blocks: usize,
) [4]f32 {
    @setFloatMode(.optimized);
    @setEvalBranchQuota(50000);
    const zero: @Vector(8, i32) = @splat(0);

    var acc: [4]f32 = .{ 0, 0, 0, 0 };

    for (0..num_super_blocks) |sb| {
        const d8 = x_d[sb];
        inline for (0..4) |row| {
            const block_ptr = group_ptr + sb * (4 * Q4K_BLOCK_BYTES) + row * Q4K_BLOCK_BYTES;
            const dw: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
            const dminw: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[2..4], .little))));
            const sm = unpackScales(block_ptr[4..16]);
            const qs = block_ptr + Q4K_HEADER_BYTES;

            var sumi: @Vector(8, i32) = zero;
            var imin: i32 = 0;
            inline for (0..4) |jp| {
                const raw: @Vector(32, u8) = (qs + jp * 32)[0..32].*;
                const lo: @Vector(32, u8) = raw & @as(@Vector(32, u8), @splat(0x0F));
                const hi: @Vector(32, u8) = raw >> @as(@Vector(32, u8), @splat(4));

                const blk_lo = sb * 8 + jp * 2;
                const blk_hi = blk_lo + 1;
                const q8_lo: @Vector(32, i8) = (x_vals + blk_lo * 32)[0..32].*;
                const q8_hi: @Vector(32, i8) = (x_vals + blk_hi * 32)[0..32].*;

                sumi += intDotU8xI8(zero, lo, q8_lo) * @as(@Vector(8, i32), @splat(@as(i32, sm.scales[jp * 2])));
                sumi += intDotU8xI8(zero, hi, q8_hi) * @as(@Vector(8, i32), @splat(@as(i32, sm.scales[jp * 2 + 1])));
                imin += @as(i32, sm.mins[jp * 2]) * x_bsums[blk_lo];
                imin += @as(i32, sm.mins[jp * 2 + 1]) * x_bsums[blk_hi];
            }

            const isum: f32 = @floatFromInt(@reduce(.Add, sumi));
            const fmin: f32 = @floatFromInt(imin);
            acc[row] += d8 * (dw * isum - dminw * fmin);
        }
    }

    return acc;
}

/// vpmaddubsw: u8 x i8 -> i16 with adjacent pairs summed (32 -> 16 lanes).
/// Saturating, but Q4_K nibbles (0..15) x Q8 (-127..127) sum to <= 3810, well
/// within i16, so no saturation occurs in practice.
inline fn maddubs(a: @Vector(32, u8), b: @Vector(32, i8)) @Vector(16, i16) {
    if (comptime builtin.cpu.arch == .x86_64 and builtin.mode != .Debug) {
        return asm ("vpmaddubsw %[b], %[a], %[out]"
            : [out] "=x" (-> @Vector(16, i16)),
            : [a] "x" (a),
              [b] "x" (b),
        );
    }
    var r: [16]i16 = undefined;
    inline for (0..16) |i| {
        const p0 = @as(i32, a[2 * i]) * @as(i32, b[2 * i]);
        const p1 = @as(i32, a[2 * i + 1]) * @as(i32, b[2 * i + 1]);
        r[i] = @intCast(@max(-32768, @min(32767, p0 + p1)));
    }
    return r;
}

/// vpmaddwd: i16 x i16 -> i32 with adjacent pairs summed (16 -> 8 lanes).
inline fn maddwd(a: @Vector(16, i16), b: @Vector(16, i16)) @Vector(8, i32) {
    if (comptime builtin.cpu.arch == .x86_64 and builtin.mode != .Debug) {
        return asm ("vpmaddwd %[b], %[a], %[out]"
            : [out] "=x" (-> @Vector(8, i32)),
            : [a] "x" (a),
              [b] "x" (b),
        );
    }
    var r: [8]i32 = undefined;
    inline for (0..8) |i| {
        r[i] = @as(i32, a[2 * i]) * @as(i32, b[2 * i]) + @as(i32, a[2 * i + 1]) * @as(i32, b[2 * i + 1]);
    }
    return r;
}

/// N-token x 4-row R4 dot with Q8_K activation. Decodes each weight block
/// (scales, nibbles, f16 deltas) ONCE per (super-block, row) and reuses it
/// across all N tokens in the tile, so the per-weight unpack work and the L1
/// weight reads amortize over N tokens instead of 2. Only N i32 accumulators
/// are live per row (not 4*N), so N can grow to 4-8 within the AVX2 register
/// file. Tokens are contiguous in the quantized buffers: token t lives at
/// `xv + t*num_sb*256`, `xd + t*num_sb`, `xb + t*num_sb*8`.
inline fn dotQ4_K_q8K_R4xN(
    comptime N: usize,
    noalias xv: [*]const i8,
    noalias xd: [*]const f32,
    noalias xb: [*]const i32,
    noalias group_ptr: [*]const u8,
    num_super_blocks: usize,
) [N][4]f32 {
    @setFloatMode(.optimized);
    @setEvalBranchQuota(100000);
    const zero: @Vector(8, i32) = @splat(0);
    const vstride = num_super_blocks * Q4K_BLOCK_SIZE;
    const dstride = num_super_blocks;
    const bstride = num_super_blocks * 8;

    var acc: [N][4]f32 = [_][4]f32{.{ 0, 0, 0, 0 }} ** N;

    for (0..num_super_blocks) |sb| {
        inline for (0..4) |row| {
            const block_ptr = group_ptr + sb * (4 * Q4K_BLOCK_BYTES) + row * Q4K_BLOCK_BYTES;
            const dw: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[0..2], .little))));
            const dminw: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_ptr[2..4], .little))));
            const sm = unpackScales(block_ptr[4..16]);
            const qs = block_ptr + Q4K_HEADER_BYTES;

            var sumi = [_]@Vector(8, i32){zero} ** N;
            var imin = [_]i32{0} ** N;

            inline for (0..4) |jp| {
                const raw: @Vector(32, u8) = (qs + jp * 32)[0..32].*;
                const lo: @Vector(32, u8) = raw & @as(@Vector(32, u8), @splat(0x0F));
                const hi: @Vector(32, u8) = raw >> @as(@Vector(32, u8), @splat(4));
                const slo: @Vector(16, i16) = @splat(@as(i16, sm.scales[jp * 2]));
                const shi: @Vector(16, i16) = @splat(@as(i16, sm.scales[jp * 2 + 1]));
                const mlo: i32 = sm.mins[jp * 2];
                const mhi: i32 = sm.mins[jp * 2 + 1];
                const blk_lo = sb * 8 + jp * 2;
                const blk_hi = blk_lo + 1;

                inline for (0..N) |t| {
                    const q8_lo: @Vector(32, i8) = (xv + t * vstride + blk_lo * 32)[0..32].*;
                    const q8_hi: @Vector(32, i8) = (xv + t * vstride + blk_hi * 32)[0..32].*;
                    // maddubs folds the nibble·q8 product to i16; maddwd applies the
                    // 6-bit weight scale AND reduces to i32 in one op (no vpmulld).
                    sumi[t] += maddwd(maddubs(lo, q8_lo), slo);
                    sumi[t] += maddwd(maddubs(hi, q8_hi), shi);
                    imin[t] += mlo * xb[t * bstride + blk_lo] + mhi * xb[t * bstride + blk_hi];
                }
            }

            inline for (0..N) |t| {
                const d8 = xd[t * dstride + sb];
                acc[t][row] += d8 * (dw * @as(f32, @floatFromInt(@reduce(.Add, sumi[t]))) - dminw * @as(f32, @floatFromInt(imin[t])));
            }
        }
    }

    return acc;
}

/// Q8_K R4 matmul: quantize f32 input (Q8_K), then dispatch the
/// integer-accumulating R4 dot. `q_d` is [batch * in/256], `q_bsums` is
/// [batch * in/256 * 8].
pub fn matmulQ8K_R4(
    pool: ?*Pool,
    input: []const f32,
    w_bytes: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
    q_vals: []i8,
    q_d: []f32,
    q_bsums: []i32,
) void {
    const num_sb = in_dim / Q4K_BLOCK_SIZE;
    for (0..batch_size) |t| {
        quantizeF32ToQ8K(
            input[t * in_dim ..][0..in_dim],
            q_vals[t * in_dim ..][0..in_dim],
            q_d[t * num_sb ..][0..num_sb],
            q_bsums[t * num_sb * 8 ..][0..num_sb * 8],
        );
    }

    const num_groups = (out_dim + 3) / 4;

    const Kernel = struct {
        fn run(
            qv: []const i8,
            qd: []const f32,
            qb: []const i32,
            out: []f32,
            weights: []const u8,
            output_dim: usize,
            batch: usize,
            n_sb: usize,
            start_group: usize,
            end_group: usize,
        ) void {
            for (start_group..end_group) |group| {
                const base_row = group * 4;
                const group_ptr = weights.ptr + group * n_sb * (4 * Q4K_BLOCK_BYTES);
                var token: usize = 0;
                while (token + Q8K_TOKEN_TILE <= batch) : (token += Q8K_TOKEN_TILE) {
                    const r = dotQ4_K_q8K_R4xN(
                        Q8K_TOKEN_TILE,
                        qv[token * n_sb * Q4K_BLOCK_SIZE ..].ptr,
                        qd[token * n_sb ..].ptr,
                        qb[token * n_sb * 8 ..].ptr,
                        group_ptr,
                        n_sb,
                    );
                    inline for (0..Q8K_TOKEN_TILE) |tt| {
                        inline for (0..4) |rr| {
                            if (base_row + rr < output_dim) out[(token + tt) * output_dim + base_row + rr] = r[tt][rr];
                        }
                    }
                }
                while (token < batch) : (token += 1) {
                    const r = dotQ4_K_q8K_R4xN(
                        1,
                        qv[token * n_sb * Q4K_BLOCK_SIZE ..].ptr,
                        qd[token * n_sb ..].ptr,
                        qb[token * n_sb * 8 ..].ptr,
                        group_ptr,
                        n_sb,
                    );
                    inline for (0..4) |rr| {
                        if (base_row + rr < output_dim) out[token * output_dim + base_row + rr] = r[0][rr];
                    }
                }
            }
        }
    };

    if (pool) |p| {
        if (num_groups >= 2) {
            const num_threads = p.threads.len + 1;
            const base = num_groups / num_threads;
            const extra = num_groups % num_threads;
            var wg: WaitGroup = .{};
            var start: usize = 0;
            for (0..num_threads) |ti| {
                const cnt = base + @intFromBool(ti < extra);
                if (cnt > 0) {
                    p.spawnWg(&wg, Kernel.run, .{ q_vals, q_d, q_bsums, output, w_bytes, out_dim, batch_size, num_sb, start, start + cnt });
                    start += cnt;
                }
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(q_vals, q_d, q_bsums, output, w_bytes, out_dim, batch_size, num_sb, 0, num_groups);
}

/// Fused gate+up (SiLU*hadamard) with Q8_K activation + R4 weights.
pub fn matmulQ8K_SiluHadamard_R4(
    pool: ?*Pool,
    input: []const f32,
    gate_weights: []const u8,
    up_weights: []const u8,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
    q_vals: []i8,
    q_d: []f32,
    q_bsums: []i32,
) void {
    const num_sb = in_dim / Q4K_BLOCK_SIZE;
    for (0..batch_size) |t| {
        quantizeF32ToQ8K(
            input[t * in_dim ..][0..in_dim],
            q_vals[t * in_dim ..][0..in_dim],
            q_d[t * num_sb ..][0..num_sb],
            q_bsums[t * num_sb * 8 ..][0..num_sb * 8],
        );
    }

    const num_groups = (out_dim + 3) / 4;

    const Kernel = struct {
        fn run(
            qv: []const i8,
            qd: []const f32,
            qb: []const i32,
            out: []f32,
            gate_w: []const u8,
            up_w: []const u8,
            output_dim: usize,
            batch: usize,
            n_sb: usize,
            start_group: usize,
            end_group: usize,
        ) void {
            for (start_group..end_group) |group| {
                const base_row = group * 4;
                const gate_ptr = gate_w.ptr + group * n_sb * (4 * Q4K_BLOCK_BYTES);
                const up_ptr = up_w.ptr + group * n_sb * (4 * Q4K_BLOCK_BYTES);
                var token: usize = 0;
                while (token + Q8K_TOKEN_TILE <= batch) : (token += Q8K_TOKEN_TILE) {
                    const v = qv[token * n_sb * Q4K_BLOCK_SIZE ..].ptr;
                    const d = qd[token * n_sb ..].ptr;
                    const b = qb[token * n_sb * 8 ..].ptr;
                    const g = dotQ4_K_q8K_R4xN(Q8K_TOKEN_TILE, v, d, b, gate_ptr, n_sb);
                    const u = dotQ4_K_q8K_R4xN(Q8K_TOKEN_TILE, v, d, b, up_ptr, n_sb);
                    inline for (0..Q8K_TOKEN_TILE) |tt| {
                        inline for (0..4) |rr| {
                            if (base_row + rr < output_dim) {
                                out[(token + tt) * output_dim + base_row + rr] = common.silu(g[tt][rr]) * u[tt][rr];
                            }
                        }
                    }
                }
                while (token < batch) : (token += 1) {
                    const v = qv[token * n_sb * Q4K_BLOCK_SIZE ..].ptr;
                    const d = qd[token * n_sb ..].ptr;
                    const b = qb[token * n_sb * 8 ..].ptr;
                    const g = dotQ4_K_q8K_R4xN(1, v, d, b, gate_ptr, n_sb);
                    const u = dotQ4_K_q8K_R4xN(1, v, d, b, up_ptr, n_sb);
                    inline for (0..4) |rr| {
                        if (base_row + rr < output_dim) out[token * output_dim + base_row + rr] = common.silu(g[0][rr]) * u[0][rr];
                    }
                }
            }
        }
    };

    if (pool) |p| {
        if (num_groups >= 2) {
            const num_threads = p.threads.len + 1;
            const base = num_groups / num_threads;
            const extra = num_groups % num_threads;
            var wg: WaitGroup = .{};
            var start: usize = 0;
            for (0..num_threads) |ti| {
                const cnt = base + @intFromBool(ti < extra);
                if (cnt > 0) {
                    p.spawnWg(&wg, Kernel.run, .{ q_vals, q_d, q_bsums, output, gate_weights, up_weights, out_dim, batch_size, num_sb, start, start + cnt });
                    start += cnt;
                }
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    Kernel.run(q_vals, q_d, q_bsums, output, gate_weights, up_weights, out_dim, batch_size, num_sb, 0, num_groups);
}

test "q8k R4 dot matches scalar dequant reference" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();

    const in_dim: usize = 512; // 2 super-blocks
    const num_sb = in_dim / Q4K_BLOCK_SIZE;

    // Random row-major Q4_K weights for 4 rows (one R4 group).
    const row_bytes = num_sb * Q4K_BLOCK_BYTES;
    const raw = try allocator.alloc(u8, 4 * row_bytes);
    defer allocator.free(raw);
    for (raw) |*b| b.* = rnd.int(u8);
    // Random bytes make the d/dmin f16 fields possibly NaN/Inf — overwrite
    // them with finite values so only the scales/nibbles stay random.
    for (0..4) |row| for (0..num_sb) |sb| {
        const off = row * row_bytes + sb * Q4K_BLOCK_BYTES;
        std.mem.writeInt(u16, raw[off..][0..2], @bitCast(@as(f16, @floatCast(rnd.float(f32) * 0.1 + 0.01))), .little);
        std.mem.writeInt(u16, raw[off + 2 ..][0..2], @bitCast(@as(f16, @floatCast(rnd.float(f32) * 0.05 + 0.01))), .little);
    };

    // Random f32 input.
    const input = try allocator.alloc(f32, in_dim);
    defer allocator.free(input);
    for (input) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;

    // Quantize input Q8_K.
    const qv = try allocator.alloc(i8, in_dim);
    defer allocator.free(qv);
    const qd = try allocator.alloc(f32, num_sb);
    defer allocator.free(qd);
    const qb = try allocator.alloc(i32, num_sb * 8);
    defer allocator.free(qb);
    quantizeF32ToQ8K(input, qv, qd, qb);

    // R4-repack weights (repackQ4KR4 frees `raw`, so dupe first).
    const raw_copy = try allocator.dupe(u8, raw);
    const packed_w = try repackQ4KR4(allocator, raw_copy, in_dim, 4);
    defer allocator.free(@constCast(packed_w));

    const simd = dotQ4_K_q8K_R4(qv.ptr, qd.ptr, qb.ptr, packed_w.ptr, num_sb);

    // Scalar reference: dequantize activation (q*d8) and weight, plain f32 dot.
    for (0..4) |row| {
        var ref: f32 = 0;
        for (0..num_sb) |sb| {
            const block = raw[row * row_bytes + sb * Q4K_BLOCK_BYTES ..][0..Q4K_BLOCK_BYTES];
            const dw: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
            const dminw: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[2..4], .little))));
            const sm = unpackScales(block[4..16]);
            const qs = block[Q4K_HEADER_BYTES..Q4K_BLOCK_BYTES];
            for (0..4) |jp| {
                const sc1: f32 = dw * @as(f32, @floatFromInt(sm.scales[jp * 2]));
                const m1: f32 = dminw * @as(f32, @floatFromInt(sm.mins[jp * 2]));
                const sc2: f32 = dw * @as(f32, @floatFromInt(sm.scales[jp * 2 + 1]));
                const m2: f32 = dminw * @as(f32, @floatFromInt(sm.mins[jp * 2 + 1]));
                for (0..32) |l| {
                    const byte = qs[jp * 32 + l];
                    const idx_lo = sb * Q4K_BLOCK_SIZE + jp * 64 + l;
                    const idx_hi = idx_lo + 32;
                    const a_lo = @as(f32, @floatFromInt(qv[idx_lo])) * qd[sb];
                    const a_hi = @as(f32, @floatFromInt(qv[idx_hi])) * qd[sb];
                    ref += a_lo * (sc1 * @as(f32, @floatFromInt(byte & 0x0F)) - m1);
                    ref += a_hi * (sc2 * @as(f32, @floatFromInt(byte >> 4)) - m2);
                }
            }
        }
        try std.testing.expectApproxEqAbs(ref, simd[row], @abs(ref) * 1e-3 + 1e-3);
    }
}

test "q8k R4xN matches per-token R4" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBADF00D);
    const rnd = prng.random();

    const N = 4;
    const in_dim: usize = 768; // 3 super-blocks
    const num_sb = in_dim / Q4K_BLOCK_SIZE;
    const row_bytes = num_sb * Q4K_BLOCK_BYTES;

    const raw = try allocator.alloc(u8, 4 * row_bytes);
    defer allocator.free(raw);
    for (raw) |*b| b.* = rnd.int(u8);
    for (0..4) |row| for (0..num_sb) |sb| {
        const off = row * row_bytes + sb * Q4K_BLOCK_BYTES;
        std.mem.writeInt(u16, raw[off..][0..2], @bitCast(@as(f16, @floatCast(rnd.float(f32) * 0.1 + 0.01))), .little);
        std.mem.writeInt(u16, raw[off + 2 ..][0..2], @bitCast(@as(f16, @floatCast(rnd.float(f32) * 0.05 + 0.01))), .little);
    };
    const packed_w = try repackQ4KR4(allocator, try allocator.dupe(u8, raw), in_dim, 4);
    defer allocator.free(@constCast(packed_w));

    // N tokens of random input, quantized contiguously.
    const qv = try allocator.alloc(i8, N * in_dim);
    defer allocator.free(qv);
    const qd = try allocator.alloc(f32, N * num_sb);
    defer allocator.free(qd);
    const qb = try allocator.alloc(i32, N * num_sb * 8);
    defer allocator.free(qb);
    for (0..N) |t| {
        const input = try allocator.alloc(f32, in_dim);
        defer allocator.free(input);
        for (input) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
        quantizeF32ToQ8K(input, qv[t * in_dim ..][0..in_dim], qd[t * num_sb ..][0..num_sb], qb[t * num_sb * 8 ..][0..num_sb * 8]);
    }

    const wide = dotQ4_K_q8K_R4xN(N, qv.ptr, qd.ptr, qb.ptr, packed_w.ptr, num_sb);
    for (0..N) |t| {
        const single = dotQ4_K_q8K_R4(
            qv[t * in_dim ..].ptr,
            qd[t * num_sb ..].ptr,
            qb[t * num_sb * 8 ..].ptr,
            packed_w.ptr,
            num_sb,
        );
        for (0..4) |row| try std.testing.expectApproxEqAbs(single[row], wide[t][row], @abs(single[row]) * 1e-4 + 1e-4);
    }
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

    if (comptime builtin.cpu.arch == .x86_64) {
        // vpsignb(a,a) = |a| and vpsignb(b,a) = sign(a) * b; cheaper
        // than @abs + @select + negate.
        const abs_a = asm ("vpsignb %[a], %[a], %[out]"
            : [out] "=x" (-> @Vector(32, u8)),
            : [a] "x" (a),
        );
        const signed_b = asm ("vpsignb %[a], %[b], %[out]"
            : [out] "=x" (-> @Vector(32, u8)),
            : [a] "x" (a),
              [b] "x" (b),
        );
        return vpdpbusd(acc, abs_a, signed_b);
    }

    // Fallback sign trick.
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
