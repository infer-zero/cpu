const std = @import("std");
const thread_pool = @import("../thread_pool.zig");

const Pool = thread_pool.Pool;
const WaitGroup = thread_pool.WaitGroup;

// ---- Element-wise operations ----

pub fn rmsNorm(state: []f32, weights: []const f32, epsilon: f32) void {
    @setFloatMode(.optimized);
    const V = @Vector(8, f32);
    const n = state.len;
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const s: V = state[i..][0..8].*;
        acc += s * s;
    }
    var sum_sq: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum_sq += state[i] * state[i];
    const mean: f32 = sum_sq / @as(f32, @floatFromInt(n));
    const inv_rms: f32 = 1.0 / @sqrt(mean + epsilon);
    const ir: V = @splat(inv_rms);
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        const s: V = state[i..][0..8].*;
        const w: V = weights[i..][0..8].*;
        state[i..][0..8].* = s * ir * w;
    }
    while (i < n) : (i += 1) state[i] = state[i] * inv_rms * weights[i];
}

/// Per-head RMSNorm: applies RMSNorm independently to each head slice.
/// `norm_weights` has length `head_dim` and is shared across all heads.
pub fn rmsNormPerHead(
    data: []f32,
    norm_weights: []const f32,
    epsilon: f32,
    head_dim: usize,
    num_heads: usize,
) void {
    @setFloatMode(.optimized);
    const V = @Vector(8, f32);
    for (0..num_heads) |head| {
        const base = head * head_dim;
        const head_slice = data[base..][0..head_dim];
        var acc: V = @splat(0);
        var i: usize = 0;
        while (i + 8 <= head_dim) : (i += 8) {
            const s: V = head_slice[i..][0..8].*;
            acc += s * s;
        }
        var sum_sq: f32 = @reduce(.Add, acc);
        while (i < head_dim) : (i += 1) sum_sq += head_slice[i] * head_slice[i];
        const mean: f32 = sum_sq / @as(f32, @floatFromInt(head_dim));
        const inv_rms: f32 = 1.0 / @sqrt(mean + epsilon);
        const ir: V = @splat(inv_rms);
        i = 0;
        while (i + 8 <= head_dim) : (i += 8) {
            const s: V = head_slice[i..][0..8].*;
            const w: V = norm_weights[i..][0..8].*;
            head_slice[i..][0..8].* = s * ir * w;
        }
        while (i < head_dim) : (i += 1) head_slice[i] = head_slice[i] * inv_rms * norm_weights[i];
    }
}

pub fn rope(
    data: []f32,
    cos: []const f32,
    sin: []const f32,
    head_dim: usize,
    num_heads: usize,
) void {
    ropePartial(data, cos, sin, head_dim, head_dim, num_heads);
}

/// Like rope, but only rotates the first `rotary_dim` elements of each head.
/// Elements beyond rotary_dim are left unchanged. When rotary_dim == head_dim,
/// this is identical to rope().
pub fn ropePartial(
    data: []f32,
    cos: []const f32,
    sin: []const f32,
    head_dim: usize,
    rotary_dim: usize,
    num_heads: usize,
) void {
    @setFloatMode(.optimized);
    const half_rotary = rotary_dim / 2;
    for (0..num_heads) |head| {
        const base = head * head_dim;
        const first_half = data[base..][0..half_rotary];
        const second_half = data[base + half_rotary ..][0..half_rotary];
        for (0..half_rotary) |dim_index| {
            const first = first_half[dim_index];
            const second = second_half[dim_index];
            first_half[dim_index] = first * cos[dim_index] - second * sin[dim_index];
            second_half[dim_index] = first * sin[dim_index] + second * cos[dim_index];
        }
    }
}

pub fn dot(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.optimized);
    var sum: f32 = 0.0;
    for (a, b) |a_val, b_val| {
        sum += a_val * b_val;
    }
    return sum;
}

pub fn scaledAdd(
    output: []f32,
    values: []const f32,
    scale: f32,
) void {
    @setFloatMode(.optimized);
    // Explicit @Vector — Zig 0.16 does not auto-vectorize this scalar loop.
    const V = @Vector(8, f32);
    const sv: V = @splat(scale);
    const n = output.len;
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const o: V = output[i..][0..8].*;
        const v: V = values[i..][0..8].*;
        output[i..][0..8].* = o + v * sv;
    }
    while (i < n) : (i += 1) output[i] += values[i] * scale;
}

/// Weighted sum of `weights.len` position vectors into `output`.
/// Equivalent to:
///   @memset(output, 0);
///   for (weights, 0..) |w, pos| {
///       for (output, 0..) |*o, i| o.* += values[pos * stride + offset + i] * w;
///   }
pub fn weightedSumF32(
    output: []f32,
    values: []const f32,
    stride: usize,
    offset: usize,
    weights: []const f32,
) void {
    @setFloatMode(.optimized);
    const dim = output.len;
    for (0..dim) |i| {
        var acc: f32 = 0;
        for (weights, 0..) |w, pos| {
            acc += values[pos * stride + offset + i] * w;
        }
        output[i] = acc;
    }
}

pub fn softmax(scores: []f32) void {
    @setFloatMode(.optimized);
    var max_val: f32 = scores[0];
    for (scores[1..]) |score| max_val = @max(max_val, score);

    var sum_exp: f32 = 0.0;
    for (scores) |*score| {
        score.* = @exp(score.* - max_val);
        sum_exp += score.*;
    }

    if (sum_exp > 0.0) {
        const inv: f32 = 1.0 / sum_exp;
        for (scores) |*score| score.* *= inv;
    }
}

/// SiLU activation: x * sigmoid(x).
/// Deliberately NOT using @setFloatMode(.optimized) — under fast-math, the
/// `nnan` flag causes the division-by-zero safety check to misfire when NaN
/// values propagate through the network (e.g. from random fuzzer inputs).
/// The SiLU is computed once per output element (outside the hot matmul inner
/// loop), so the performance impact of strict float mode here is negligible.
pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn addVectors(a: []f32, b: []const f32) void {
    @setFloatMode(.optimized);
    const V = @Vector(8, f32);
    const n = a.len;
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const av: V = a[i..][0..8].*;
        const bv: V = b[i..][0..8].*;
        a[i..][0..8].* = av + bv;
    }
    while (i < n) : (i += 1) a[i] += b[i];
}

pub fn scaleVector(data: []f32, scale: f32) void {
    @setFloatMode(.optimized);
    for (data) |*val| {
        val.* *= scale;
    }
}

// ---- Threaded matmul ----

/// Matmul with F32 weights (for tied embeddings), parallelized.
pub fn matmulF32(
    pool: ?*Pool,
    input: []const f32,
    weights: []const f32,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
) void {
    const Kernel = struct {
        fn run(
            batch_input: []const f32,
            batch_output: []f32,
            weight_data: []const f32,
            input_dim: usize,
            output_dim: usize,
            batch: usize,
            start_row: usize,
            end_row: usize,
        ) void {
            @setFloatMode(.optimized);
            for (start_row..end_row) |row| {
                const weight_row = weight_data[row * input_dim ..].ptr;
                for (0..batch) |token| {
                    batch_output[token * output_dim + row] = dotF32(weight_row, batch_input[token * input_dim ..].ptr, input_dim);
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
                        input,
                        output,
                        weights,
                        in_dim,
                        out_dim,
                        batch_size,
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
    Kernel.run(input, output, weights, in_dim, out_dim, batch_size, 0, out_dim);
}

inline fn dotF32(noalias a: [*]const f32, noalias b: [*]const f32, len: usize) f32 {
    @setFloatMode(.optimized);
    // Explicit @Vector — Zig 0.16 does not auto-vectorize this reduction.
    const V = @Vector(8, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const av: V = (a + i)[0..8].*;
        const bv: V = (b + i)[0..8].*;
        acc += av * bv;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < len) : (i += 1) sum += a[i] * b[i];
    return sum;
}
