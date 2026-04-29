const std = @import("std");

pub const Info = struct {
    model_mib: u64,
    available_mib: u64,
    limit_mib: u64, // 90% of available
    fit: bool,
};

/// Lazily resolve a default `Io` for the simple, blocking probes below.
/// `mem.zig` is invoked from leaf code that does not currently have access
/// to the upstream `Io`, so we fall back to the stdlib's global single-
/// threaded instance — adequate for one-off file-stat operations.
fn defaultIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

pub fn checkModelFitsInMemory(path: []const u8) ?Info {
    const model_bytes = estimateModelSize(path) orelse return null;
    const available_bytes = getAvailableMemory() orelse return null;

    const model_mib = model_bytes / (1024 * 1024);
    const available_mib = available_bytes / (1024 * 1024);
    const limit_mib = available_mib / 10 * 9; // 90%

    return .{
        .model_mib = model_mib,
        .available_mib = available_mib,
        .limit_mib = limit_mib,
        .fit = model_mib < limit_mib,
    };
}

pub fn estimateModelSize(path: []const u8) ?u64 {
    const io = defaultIo();
    const cwd = std.Io.Dir.cwd();

    // Try as single file (GGUF)
    if (cwd.openFile(io, path, .{})) |file| {
        defer file.close(io);
        const stat = file.stat(io) catch return null;
        if (stat.kind == .file) {
            return if (stat.size > 0) stat.size else null;
        }
    } else |_| {}

    // Try as directory (HuggingFace with .safetensors files)
    var dir = cwd.openDir(io, path, .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var total: u64 = 0;
    var iter = dir.iterate();
    while (iter.next(io) catch return null) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.name, ".safetensors")) {
            const file = dir.openFile(io, entry.name, .{}) catch continue;
            defer file.close(io);
            const stat = file.stat(io) catch continue;
            total += stat.size;
        }
    }

    return if (total > 0) total else null;
}

pub fn getAvailableMemory() ?u64 {
    const builtin = @import("builtin");
    if (builtin.os.tag != .linux) return null;

    const io = defaultIo();
    const file = std.Io.Dir.openFileAbsolute(io, "/proc/meminfo", .{}) catch return null;
    defer file.close(io);

    var buf: [4096]u8 = undefined;
    const len = file.readPositional(io, &.{&buf}, 0) catch return null;
    const content = buf[0..len];

    const needle = "MemAvailable:";
    const start = std.mem.find(u8, content, needle) orelse return null;
    const rest = content[start + needle.len ..];

    // Skip whitespace
    var i: usize = 0;
    while (i < rest.len and rest[i] == ' ') : (i += 1) {}

    // Parse the number (value is in kB)
    var end = i;
    while (end < rest.len and rest[end] >= '0' and rest[end] <= '9') : (end += 1) {}

    const kb = std.fmt.parseInt(u64, rest[i..end], 10) catch return null;
    return kb * 1024; // convert kB to bytes
}

test "estimateModelSize returns null for nonexistent path" {
    const result = estimateModelSize("./nonexistent_model_path_that_does_not_exist");
    try std.testing.expectEqual(null, result);
}

test "getAvailableMemory returns value on linux" {
    const builtin = @import("builtin");
    if (builtin.os.tag == .linux) {
        const m = getAvailableMemory();
        try std.testing.expect(m != null);
        try std.testing.expect(m.? > 0);
    }
}
