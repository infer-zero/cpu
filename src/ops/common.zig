const std = @import("std");

const Pool = std.Thread.Pool;
const WaitGroup = std.Thread.WaitGroup;

// ---- Element-wise operations ----

pub fn rmsNorm(state: []f32, weights: []const f32, epsilon: f32) void {
    // Accumulate sum-of-squares in f64 to match llama.cpp's `ggml_float`
    // (typedef'd to double). f32 accumulation drifts enough across many
    // layers to flip greedy argmax after ~25-30 decode tokens on
    // Qwen3-0.6B-Q8_0; doubling precision here matches llama.cpp's
    // logit ordering.
    var sum_sq: f64 = 0.0;
    {
        @setFloatMode(.optimized);
        for (state) |value| sum_sq += @as(f64, value) * @as(f64, value);
    }
    const mean: f64 = sum_sq / @as(f64, @floatFromInt(state.len));
    const inv_rms: f32 = @floatCast(1.0 / @sqrt(mean + @as(f64, epsilon)));
    {
        @setFloatMode(.optimized);
        for (state, weights) |*state_val, weight_val| {
            state_val.* = state_val.* * inv_rms * weight_val;
        }
    }
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
    for (0..num_heads) |head| {
        const base = head * head_dim;
        const head_slice = data[base..][0..head_dim];
        var sum_sq: f64 = 0.0;
        for (head_slice) |value| sum_sq += @as(f64, value) * @as(f64, value);
        const mean: f64 = sum_sq / @as(f64, @floatFromInt(head_dim));
        const inv_rms: f32 = @floatCast(1.0 / @sqrt(mean + @as(f64, epsilon)));
        for (head_slice, norm_weights[0..head_dim]) |*sv, wv| {
            sv.* = sv.* * inv_rms * wv;
        }
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
    // f64 accumulator: serial f32 sum over head_dim values drifts enough
    // across many attention scoring calls to flip argmax. Llama.cpp uses
    // SIMD-parallel f32 (~log(N) error); f64 here matches or exceeds that.
    var sum: f64 = 0.0;
    for (a, b) |a_val, b_val| {
        sum += @as(f64, a_val) * @as(f64, b_val);
    }
    return @floatCast(sum);
}

pub fn scaledAdd(
    output: []f32,
    values: []const f32,
    scale: f32,
) void {
    @setFloatMode(.optimized);
    for (output, values) |*out_val, val| {
        out_val.* += val * scale;
    }
}

/// Weighted sum of `weights.len` position vectors into `output`.
/// Equivalent to:
///   @memset(output, 0);
///   for (weights, 0..) |w, pos| {
///       for (output, 0..) |*o, i| o.* += values[pos * stride + offset + i] * w;
///   }
/// but with the inner sum accumulated in f64 per element — avoids the
/// f32 drift that builds up across many attended positions in attention
/// V·weights. Mirrors the `dot`/`rmsNorm`/`softmax` f64 accumulator
/// pattern used elsewhere in this file.
pub fn weightedSumF32(
    output: []f32,
    values: []const f32,
    stride: usize,
    offset: usize,
    weights: []const f32,
) void {
    const dim = output.len;
    for (0..dim) |i| {
        var acc: f64 = 0;
        for (weights, 0..) |w, pos| {
            acc += @as(f64, values[pos * stride + offset + i]) * @as(f64, w);
        }
        output[i] = @floatCast(acc);
    }
}

pub fn softmax(scores: []f32) void {
    @setFloatMode(.optimized);
    var max_val: f32 = scores[0];
    for (scores[1..]) |score| max_val = @max(max_val, score);

    // Accumulate in f64 to match llama.cpp's `ggml_float` (double) softmax
    // sum — matters for long contexts where many small exp() values sum
    // and f32 rounding flips attention weights enough to drift logits.
    var sum_exp: f64 = 0.0;
    for (scores) |*score| {
        score.* = @exp(score.* - max_val);
        sum_exp += @as(f64, score.*);
    }

    if (sum_exp > 0.0) {
        const inv: f32 = @floatCast(1.0 / sum_exp);
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
    for (a, b) |*a_val, b_val| {
        a_val.* += b_val;
    }
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
    // f64 accumulator — used for LM head and MOE router matmuls. See
    // `dot` above.
    var sum: f64 = 0.0;
    for (0..len) |i| sum += @as(f64, a[i]) * @as(f64, b[i]);
    return @floatCast(sum);
}
