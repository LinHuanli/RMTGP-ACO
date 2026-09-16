"""资源监控只分配未领取且已放行任务，不改变实验协议。"""
from control_experiments.mmas_ls import watch
from control_experiments.mmas_ls.campaign import parse_devices
from control_experiments.mmas_ls.common import atomic_json


def test_allowed_idle_models_only():
    output="""IDLE cuda02 1 0.0 / 24.0 0% 0 - NVIDIA RTX A5000
IDLE cuda-small1 1 0.0 / 20.0 0% 0 - NVIDIA RTX 4000 Ada Generation
IDLE cuda-small0 3 0.0 / 16.0 0% 0 - NVIDIA RTX A4000
SINGLE cuda04 0 0.0 / 24.0 0% 1 other NVIDIA RTX A5000
IDLE cuda14 0 0.0 / 48.0 0% 0 - NVIDIA A40
IDLE cuda09 1 0.0 / 24.0 0% 0 - NVIDIA RTX PRO 4000 Blackwell
IDLE cuda22 0 0.0 / 48.0 0% 0 - NVIDIA RTX 6000 Ada Generation
IDLE cuda02 1 0.0 / 24.0 0% 0 - NVIDIA RTX A5000
"""
    assert parse_devices(output)==[("cuda02",1),("cuda-small1",1),("cuda-small0",3)]


def test_runnable_tasks_preserve_gates_locks_and_attempts(tmp_path,monkeypatch):
    monkeypatch.setattr(watch,"eligible",lambda task,out:task["allowed"])
    for name,allowed in (("ready",True),("gated",False),("done",True),("claimed",True),("failed",True)):
        atomic_json(tmp_path/"queue"/(name+".json"),{"task":{"id":name,"allowed":allowed}})
    atomic_json(tmp_path/"jobs/done/status.json",{"status":"completed"})
    atomic_json(tmp_path/"jobs/failed/attempts.json",{"count":3})
    (tmp_path/"locks/tasks/claimed.lock.d").mkdir(parents=True)
    assert watch.runnable_tasks(tmp_path)==["ready"]


def test_scan_limits_launches_and_cools_down(tmp_path,monkeypatch):
    monkeypatch.setattr(watch,"devices",lambda:([("cuda02",1),("cuda-small1",1)],"test listing"))
    monkeypatch.setattr(watch,"runnable_tasks",lambda out:["one"])
    launched=[]
    def launch(host,gpu,out):
        launched.append((host,gpu));return {"host":host,"gpu":gpu}
    monkeypatch.setattr(watch,"launch_device",launch)
    last={};watch.scan(tmp_path,last);watch.scan(tmp_path,last);watch.scan(tmp_path,last)
    assert launched==[("cuda02",1),("cuda-small1",1)]
    monkeypatch.setattr(watch,"runnable_tasks",lambda out:[])
    result=watch.scan(tmp_path,{})
    assert not result["launched"]


def test_launch_failure_does_not_stop_scan(tmp_path,monkeypatch):
    monkeypatch.setattr(watch,"devices",lambda:([("bad-host",0),("good-host",0)],"test listing"))
    monkeypatch.setattr(watch,"runnable_tasks",lambda out:["one"])
    def launch(host,gpu,out):
        if host=="bad-host":raise OSError("unreachable")
        return {"host":host,"gpu":gpu}
    monkeypatch.setattr(watch,"launch_device",launch)
    result=watch.scan(tmp_path,{})
    assert len(result["errors"])==1 and len(result["launched"])==1


def test_scan_does_not_launch_surplus_workers_for_another_model(tmp_path,monkeypatch):
    listing="\n".join(f"IDLE cuda0{i} 0 0.0 / 24.0 0% 0 - NVIDIA RTX A5000" for i in range(3))
    monkeypatch.setattr(watch,"devices",lambda:([(f"cuda0{i}",0) for i in range(3)],listing))
    monkeypatch.setattr(watch,"runnable_tasks",lambda out,model=None:["one"] if model else ["one","waiting-a4000","waiting-ada"])
    monkeypatch.setattr(watch,"launch_device",lambda host,gpu,out:{"host":host,"gpu":gpu})
    result=watch.scan(tmp_path,{})
    assert len(result["launched"])==1
