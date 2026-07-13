# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 资源监控线程 —— 监控训练过程中的CPU/IO使用情况
# ============================================================================
# 这个模块提供了一个后台线程，用于定期采样进程的资源使用情况。
# 主要用于大规模分布式训练时监控系统健康状态。
#
# 监控指标包括：
# - CPU使用率
# - IO读写次数和字节数
# - CPU时间分配（用户态/系统态/子进程/IO等待）
# - CPU亲和性（进程绑定在哪些CPU核心上）
# - 线程数
# - 上下文切换次数

import dataclasses
import threading
from typing import Dict, Tuple
import time

import psutil  # 系统和进程监控库


@dataclasses.dataclass
class ResourceStatsSample:
    """
    资源统计数据结构

    使用Python dataclass自动生成__init__等方法。
    每个字段对应一种资源监控指标。
    """
    timestamp: float                    # 采样时间戳
    cpu_percent: float                  # CPU使用率（百分比）
    read_count: int                     # 磁盘读操作次数
    write_count: int                    # 磁盘写操作次数
    read_bytes: int                     # 磁盘读字节数
    write_bytes: int                    # 磁盘写字节数
    read_chars: int                     # 终端读字符数
    write_chars: int                    # 终端写字符数
    cpu_times_user: float               # 用户态CPU时间
    cpu_times_system: float             # 系统态CPU时间
    cpu_times_children_user: float      # 子进程用户态CPU时间
    cpu_times_children_system: float    # 子进程系统态CPU时间
    cpu_times_iowait: float             # IO等待CPU时间
    cpu_affinity: str                   # CPU亲和性（绑定在哪些核心）
    cpu_num: int                        # 当前运行的CPU编号
    num_threads: int                    # 线程数
    num_voluntary_ctx_switches: int     # 主动上下文切换次数
    num_involuntary_ctx_switches: int   # 被动上下文切换次数（时间片用完）

    def as_tuple(self) -> Dict:
        """以tuple形式返回所有字段值"""
        return dataclasses.astuple(self)

    def fields(self) -> Tuple[dataclasses.Field, ...]:
        """返回dataclass的字段定义"""
        return dataclasses.fields(self.__class__)


class ResourceMonitoringThread(threading.Thread):
    """
    资源监控后台线程

    每隔一定时间采样进程的资源使用情况，并通过回调函数报告。
    运行在独立线程中，不影响主训练循环的性能。
    """

    def __init__(self, pid=None, refresh_interval=None, stats_callback_fn=None):
        """
        启动资源监控线程

        参数:
            pid: 要监控的进程ID（默认当前进程）
            refresh_interval: 采样间隔（秒，默认5秒）
            stats_callback_fn: 回调函数，每采样一次调用一次
        """
        super(ResourceMonitoringThread, self).__init__()
        if refresh_interval is None:
            refresh_interval = 5
        self.is_running_event = threading.Event()  # 用于控制线程启停
        self.p = psutil.Process(pid)               # 要监控的进程
        self.refresh_interval = refresh_interval
        if stats_callback_fn is None:
            # 默认回调：打印统计信息
            def stats_callback_fn(resource_sample: ResourceStatsSample):
                print(
                    f"PID {self.p.pid} Stats: {resource_sample.resource_stats}")
        elif not callable(stats_callback_fn):
            raise ValueError("Callback needs to be callable, got {}".format(
                type(stats_callback_fn)))
        self.stats_callback_fn = stats_callback_fn

    def stop(self) -> None:
        """停止监控线程"""
        self.is_running_event.set()

    def run(self) -> None:
        """线程主循环：定期采样资源使用情况"""
        while not self.is_running_event.is_set():
            self.sample_counters()
            self.is_running_event.wait(self.refresh_interval)  # 等待下一次采样

    def log_sample(self, resource_sample: ResourceStatsSample) -> None:
        """调用回调函数处理采样数据"""
        self.stats_callback_fn(resource_sample)

    def sample_counters(self) -> None:
        """
        采样当前的资源使用情况

        使用psutil的oneshot()上下文管理器提高效率：
        在oneshot内部，所有psutil调用共享一次系统调用，
        避免反复切换内核态。
        """
        if not self.p.is_running():
            self.stop()
            return

        with self.p.oneshot():  # 高效批量采样
            cpu_percent = self.p.cpu_percent()
            cpu_times = self.p.cpu_times()
            io_counters = self.p.io_counters()
            cpu_affinity = self.p.cpu_affinity()
            cpu_num = self.p.cpu_num()
            num_threads = self.p.num_threads()
            num_ctx_switches = self.p.num_ctx_switches()
        timestamp = time.time()

        read_count = io_counters.read_count
        write_count = io_counters.write_count
        read_bytes = io_counters.read_bytes
        write_bytes = io_counters.write_bytes
        read_chars = io_counters.read_chars
        write_chars = io_counters.write_chars

        def compress_cpu_affinity(cpu_affinity):
            """
            将CPU亲和性列表压缩为区间表示

            例如：[0, 1, 2, 3, 5, 6, 7] → "0-3,5-7"
            这样更易于阅读和存储。
            """
            if not cpu_affinity:
                return ""
            cpu_affinity_compressed = []
            min_x = None
            max_x = None
            last_x = None

            for x in cpu_affinity:
                if last_x is None:
                    min_x = x
                    max_x = x
                    last_x = x
                    continue
                elif x == (last_x + 1):
                    max_x = x
                elif max_x is not None:
                    if min_x == max_x:
                        cpu_affinity_compressed.append("{}".format(min_x))
                    else:
                        cpu_affinity_compressed.append("{}-{}".format(min_x, max_x))
                    min_x = x
                    max_x = x
                last_x = x

            if max_x is not None:
                if min_x == max_x:
                    cpu_affinity_compressed.append("{}".format(min_x))
                else:
                    cpu_affinity_compressed.append("{}-{}".format(min_x, max_x))

            cpu_affinity_compressed = ",".join(cpu_affinity_compressed)
            return cpu_affinity_compressed

        cpu_affinity = compress_cpu_affinity(cpu_affinity)

        # 构造采样数据并回调
        resource_sample = ResourceStatsSample(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            read_count=read_count,
            write_count=write_count,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            read_chars=read_chars,
            write_chars=write_chars,
            cpu_times_user=cpu_times.user,
            cpu_times_system=cpu_times.system,
            cpu_times_children_user=cpu_times.children_user,
            cpu_times_children_system=cpu_times.children_system,
            cpu_times_iowait=cpu_times.iowait,
            cpu_affinity=cpu_affinity,
            cpu_num=cpu_num,
            num_threads=num_threads,
            num_voluntary_ctx_switches=num_ctx_switches.voluntary,
            num_involuntary_ctx_switches=num_ctx_switches.involuntary,
        )
        self.log_sample(resource_sample)


if __name__ == "__main__":
    # 测试：监控5秒
    import multiprocessing
    pid = multiprocessing.current_process().pid
    monitor_thread = ResourceMonitoringThread(pid, 1)
    monitor_thread.start()
    time.sleep(5)
    print("Shutdown")
    monitor_thread.stop()
