"""汇集验收证据。生成报告不等于自动批准确认性实验。"""
from __future__ import annotations
from pathlib import Path
from .common import OUT,atomic_json,read_json,now
from .campaign import status
from .precision import report as precision_report
from .reproduction import compare


def collect(out=OUT):
    out=Path(out)
    report={"time":now(),"queue":status(out),"native":read_json(out/"validation/native.json",{}),
        "data_provenance":{k:v for k,v in read_json(out/"validation/data_provenance.json",{}).items()
                           if k in ("status","errors","overlap","scope","hashes","instance_ids")},
        "precision":precision_report(out),"reproduction":compare(out),
        "formal_gate":"not automatically approved; inspect numeric and historical component effects before P1/P2"}
    atomic_json(out/"reports/p0_review.json",report)
    return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUT);a=p.parse_args()
    result=collect(a.output);print(result["queue"]);print("precision:",result["precision"]["status"]);print("reproduction:",result["reproduction"]["status"])
