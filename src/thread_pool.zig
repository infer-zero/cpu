const std = @import("std");
const builtin = @import("builtin");

/// Thread pool used by the CPU matmul kernels. Faithfully ports
/// `std.Thread.Pool` from Zig 0.15.2 onto the 0.16 sync primitives.
///
/// In 0.16 `std.Thread.Pool` was removed and `std.Thread.Mutex`/`Condition`/
/// `ResetEvent` moved under `std.Io`, requiring an `Io` argument. Because
/// `infer_cpu` is a leaf package that does not currently receive an `Io`
/// from upstream, we use `std.Io.Threaded.global_single_threaded` for the
/// internal sync primitives — the underlying `futexWait`/`futexWake` are
/// raw OS syscalls and work correctly across threads regardless of the
/// Io's task-scheduling mode (the "single-threaded" namespace refers to
/// async tasks, not to kernel mutexes).
///
/// Public surface mirrors the 0.15.2 Pool/WaitGroup we depended on:
///   * `Pool.threads.len`            — number of background workers
///   * `Pool.spawnWg(wg, fn, args)`  — enqueue a task, bumping the wg
///   * `Pool.waitAndWork(wg)`        — drain the queue from the main
///                                     thread, then block until workers
///                                     finish remaining tasks
///   * `WaitGroup.start/finish/wait/reset/isDone`
const sync_io = std.Io.Threaded.global_single_threaded.io();

const is_waiting: usize = 1 << 0;
const one_pending: usize = 1 << 1;

pub const WaitGroup = struct {
    state: std.atomic.Value(usize) = .init(0),
    event: std.Io.Event = .unset,

    pub fn start(self: *WaitGroup) void {
        const state = self.state.fetchAdd(one_pending, .monotonic);
        std.debug.assert((state / one_pending) < (std.math.maxInt(usize) / one_pending));
    }

    pub fn finish(self: *WaitGroup) void {
        const state = self.state.fetchSub(one_pending, .acq_rel);
        std.debug.assert((state / one_pending) > 0);

        if (state == (one_pending | is_waiting)) {
            self.event.set(sync_io);
        }
    }

    pub fn wait(self: *WaitGroup) void {
        const state = self.state.fetchAdd(is_waiting, .acquire);
        std.debug.assert(state & is_waiting == 0);

        if ((state / one_pending) > 0) {
            self.event.waitUncancelable(sync_io);
        }
    }

    pub fn reset(self: *WaitGroup) void {
        self.state.store(0, .monotonic);
        self.event.reset();
    }

    pub fn isDone(self: *WaitGroup) bool {
        const state = self.state.load(.acquire);
        // Note: 0.15.2 asserts `state & is_waiting == 0` here, but we want
        // `isDone` to be safe to poll from `waitAndWork` while the main
        // thread is also working tasks. Drop the assert; the value remains
        // correct because pending counter is independent of the wait bit.
        return (state / one_pending) == 0;
    }
};

const Runnable = struct {
    runFn: RunProto,
    node: std.SinglyLinkedList.Node = .{},
};

const RunProto = *const fn (*Runnable) void;

pub const Pool = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    run_queue: std.SinglyLinkedList = .{},
    is_running: bool = true,
    allocator: std.mem.Allocator,
    threads: []std.Thread,

    pub const Options = struct {
        allocator: std.mem.Allocator,
        n_jobs: usize,
        stack_size: usize = std.Thread.SpawnConfig.default_stack_size,
    };

    pub fn init(pool: *Pool, options: Options) !void {
        pool.* = .{
            .allocator = options.allocator,
            .threads = &.{},
        };
        if (builtin.single_threaded or options.n_jobs == 0) return;

        pool.threads = try options.allocator.alloc(std.Thread, options.n_jobs);
        var spawned: usize = 0;
        errdefer pool.join(spawned);

        for (pool.threads) |*thread| {
            thread.* = try std.Thread.spawn(.{
                .stack_size = options.stack_size,
                .allocator = options.allocator,
            }, worker, .{pool});
            spawned += 1;
        }
    }

    pub fn deinit(pool: *Pool) void {
        pool.join(pool.threads.len);
        pool.* = undefined;
    }

    fn join(pool: *Pool, spawned: usize) void {
        if (builtin.single_threaded or pool.threads.len == 0) {
            if (pool.threads.len != 0) pool.allocator.free(pool.threads);
            return;
        }

        {
            pool.mutex.lockUncancelable(sync_io);
            defer pool.mutex.unlock(sync_io);
            pool.is_running = false;
        }

        pool.cond.broadcast(sync_io);
        for (pool.threads[0..spawned]) |thread| thread.join();
        pool.allocator.free(pool.threads);
    }

    /// Queue `func(args...)` and arrange for `wait_group.finish()` to be
    /// called after it returns. Falls back to running inline if allocation
    /// fails or the build is single-threaded.
    pub fn spawnWg(pool: *Pool, wait_group: *WaitGroup, comptime func: anytype, args: anytype) void {
        wait_group.start();

        if (builtin.single_threaded or pool.threads.len == 0) {
            @call(.auto, func, args);
            wait_group.finish();
            return;
        }

        const Args = @TypeOf(args);
        const Closure = struct {
            arguments: Args,
            pool: *Pool,
            runnable: Runnable = .{ .runFn = runFn },
            wait_group: *WaitGroup,

            fn runFn(runnable: *Runnable) void {
                const closure: *@This() = @alignCast(@fieldParentPtr("runnable", runnable));
                @call(.auto, func, closure.arguments);
                closure.wait_group.finish();

                // The thread pool's allocator is protected by the mutex.
                const mutex = &closure.pool.mutex;
                mutex.lockUncancelable(sync_io);
                defer mutex.unlock(sync_io);
                closure.pool.allocator.destroy(closure);
            }
        };

        {
            pool.mutex.lockUncancelable(sync_io);

            const closure = pool.allocator.create(Closure) catch {
                pool.mutex.unlock(sync_io);
                @call(.auto, func, args);
                wait_group.finish();
                return;
            };
            closure.* = .{
                .arguments = args,
                .pool = pool,
                .wait_group = wait_group,
            };

            pool.run_queue.prepend(&closure.runnable.node);
            pool.mutex.unlock(sync_io);
        }

        // Notify outside the lock to keep the critical section small.
        pool.cond.signal(sync_io);
    }

    fn worker(pool: *Pool) void {
        pool.mutex.lockUncancelable(sync_io);
        defer pool.mutex.unlock(sync_io);

        while (true) {
            while (pool.run_queue.popFirst()) |run_node| {
                pool.mutex.unlock(sync_io);
                defer pool.mutex.lockUncancelable(sync_io);

                const runnable: *Runnable = @fieldParentPtr("node", run_node);
                runnable.runFn(runnable);
            }

            if (pool.is_running) {
                pool.cond.waitUncancelable(sync_io, &pool.mutex);
            } else {
                break;
            }
        }
    }

    /// Run pending tasks from the queue on the calling thread, then block
    /// until `wait_group` reaches zero. Mirrors `std.Thread.Pool.waitAndWork`.
    pub fn waitAndWork(pool: *Pool, wait_group: *WaitGroup) void {
        while (!wait_group.isDone()) {
            pool.mutex.lockUncancelable(sync_io);
            if (pool.run_queue.popFirst()) |run_node| {
                pool.mutex.unlock(sync_io);
                const runnable: *Runnable = @fieldParentPtr("node", run_node);
                runnable.runFn(runnable);
                continue;
            }

            pool.mutex.unlock(sync_io);
            wait_group.wait();
            return;
        }
    }
};

/// Initialize a thread pool with P-core detection on Linux. Returns null
/// if the pool cannot be created or if only one core is available.
///
/// `n_jobs_override` lets the caller force a specific worker count
/// (typically resolved from `INFER_THREADS` upstream, where libc and
/// process env are available). `null` means auto-detect via P-core
/// detection (Linux) or `getCpuCount() - 1` (other platforms).
pub fn initPool(allocator: std.mem.Allocator, n_jobs_override: ?usize) ?*Pool {
    const detected = if (comptime builtin.os.tag == .linux)
        detectAndPinPhysicalCores() orelse fallbackWorkerCount()
    else
        fallbackWorkerCount();

    const n_workers = n_jobs_override orelse detected;
    if (n_workers == 0) return null;

    const pool = allocator.create(Pool) catch return null;
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
    const file = std.Io.Dir.openFileAbsolute(sync_io, path, .{}) catch return null;
    defer file.close(sync_io);
    var buf: [32]u8 = undefined;
    const len = file.readPositional(sync_io, &.{&buf}, 0) catch return null;
    const trimmed = std.mem.trimEnd(u8, buf[0..len], &.{ '\n', ' ', '\r' });
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
