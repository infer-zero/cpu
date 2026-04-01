const std = @import("std");
const builtin = @import("builtin");

/// Initialize a thread pool with P-core detection on Linux.
/// Returns null if the pool cannot be created or if only one core is available.
pub fn initPool(allocator: std.mem.Allocator) ?*std.Thread.Pool {
    const n_workers = if (comptime builtin.os.tag == .linux)
        detectAndPinPhysicalCores() orelse fallbackWorkerCount()
    else
        fallbackWorkerCount();

    if (n_workers == 0) return null;

    const pool = allocator.create(std.Thread.Pool) catch return null;
    pool.init(.{ .allocator = allocator, .n_jobs = n_workers }) catch {
        allocator.destroy(pool);
        return null;
    };
    return pool;
}

fn fallbackWorkerCount() usize {
    const cpu_count = std.Thread.getCpuCount() catch return 0;
    return cpu_count -| 1;
}

fn readSysfsUsize(comptime fmt: []const u8, args: anytype) ?usize {
    var path_buf: [128]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, fmt, args) catch return null;
    const file = std.fs.openFileAbsolute(path, .{}) catch return null;
    defer file.close();
    var buf: [32]u8 = undefined;
    const len = file.read(&buf) catch return null;
    const trimmed = std.mem.trimRight(u8, buf[0..len], &.{ '\n', ' ', '\r' });
    return std.fmt.parseInt(usize, trimmed, 10) catch null;
}

fn detectAndPinPhysicalCores() ?usize {
    const max_cpus = 256;
    const bits_per_word = @sizeOf(usize) * 8;
    const p_core_freq_threshold_percent = 85;

    var core_keys: [max_cpus]u64 = undefined;
    var core_cpus: [max_cpus]usize = undefined;
    var core_max_freq: [max_cpus]usize = undefined;
    var num_cores: usize = 0;

    for (0..max_cpus) |cpu_id| {
        const core_id = readSysfsUsize(
            "/sys/devices/system/cpu/cpu{d}/topology/core_id",
            .{cpu_id},
        ) orelse break;
        const pkg_id = readSysfsUsize(
            "/sys/devices/system/cpu/cpu{d}/topology/physical_package_id",
            .{cpu_id},
        ) orelse 0;

        const key: u64 = (@as(u64, pkg_id) << 32) | @as(u64, core_id);

        var found = false;
        for (core_keys[0..num_cores]) |existing_key| {
            if (existing_key == key) {
                found = true;
                break;
            }
        }
        if (!found and num_cores < max_cpus) {
            core_keys[num_cores] = key;
            core_cpus[num_cores] = cpu_id;
            core_max_freq[num_cores] = readSysfsUsize(
                "/sys/devices/system/cpu/cpu{d}/cpufreq/cpuinfo_max_freq",
                .{cpu_id},
            ) orelse 0;
            num_cores += 1;
        }
    }

    if (num_cores < 2) return null;

    var max_freq: usize = 0;
    for (core_max_freq[0..num_cores]) |freq| {
        max_freq = @max(max_freq, freq);
    }

    var selected_cpus: [max_cpus]usize = undefined;
    var num_selected: usize = 0;

    if (max_freq > 0) {
        const threshold = max_freq * p_core_freq_threshold_percent / 100;
        for (0..num_cores) |core| {
            if (core_max_freq[core] >= threshold) {
                selected_cpus[num_selected] = core_cpus[core];
                num_selected += 1;
            }
        }
    }

    if (num_selected < 2) {
        num_selected = num_cores;
        @memcpy(selected_cpus[0..num_cores], core_cpus[0..num_cores]);
    }

    var mask = std.mem.zeroes(std.os.linux.cpu_set_t);
    for (selected_cpus[0..num_selected]) |cpu_id| {
        const word = cpu_id / bits_per_word;
        const bit: std.math.Log2Int(usize) = @intCast(cpu_id % bits_per_word);
        mask[word] |= @as(usize, 1) << bit;
    }

    std.os.linux.sched_setaffinity(0, &mask) catch {};

    return num_selected -| 1;
}
