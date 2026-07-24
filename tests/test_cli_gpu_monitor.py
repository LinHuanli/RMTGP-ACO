"""命令行 GPU 监控的设备作用域回归测试。"""

from __future__ import annotations

from subprocess import CompletedProcess

from rmtgp_aco import cli


def test_external_gpu_processes_only_reports_target_devices(
    monkeypatch,
) -> None:
    """其他 GPU 的作业和当前进程都不能污染目标卡 benchmark。"""

    def fake_run(command, **_kwargs):
        if "--query-gpu=index,uuid" in command:
            output = "0, GPU-A\n1, GPU-B\n2, GPU-C\n"
        else:
            output = (
                "GPU-A, 111, own-python, 286\n"
                "GPU-B, 222, competing-python, 1024\n"
                "GPU-C, 333, unrelated-python, 2048\n"
            )
        return CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    observed = cli._external_gpu_processes({0, 1}, own_pid=111)

    assert observed == [
        {
            "device": 1,
            "pid": 222,
            "process_name": "competing-python",
            "used_memory_mib": "1024",
        }
    ]
