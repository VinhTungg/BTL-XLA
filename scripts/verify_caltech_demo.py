"""Run the real demo on all 505 validation images; never open test or train."""
import argparse
import json
from pathlib import Path

from tacs.showcase.__main__ import create_engine,make_report,validation_input,write_report
from tacs.showcase.caltech_artifacts import CHECKPOINTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=505)
    args=parser.parse_args()
    if not 1<=args.limit<=505: parser.error('limit must be in 1..505')
    try:
        args.output.mkdir(parents=True,exist_ok=False)
        engine=create_engine('full100',dataset='caltech101')
        before=[p._version for model in (engine.model,engine.control) for p in model.parameters()]
        counts={}
        cases={}
        rows=[]
        for position in range(args.limit):
            image,query=validation_input(engine,position)
            report=make_report(engine,image,query)
            for name,item in report['results'].items():
                counts[name]=counts.get(name,0)+int(item['correct'])
                if (len(item['top5'])!=5 or not all(0<=x['probability']<=1 for x in item['top5'])
                        or (item['context_index'] is not None and item['context_index'] not in engine.split.candidate_pool)):
                    raise ValueError('Invalid class probability/context role')
            if (report['metadata']['checkpoint_sha256']!=CHECKPOINTS['two_stage']
                    or report['metadata']['weights_updated'] is not False
                    or report['metadata']['test_evaluated'] is not False):
                raise ValueError('Unexpected checkpoint/provenance')
            selected=report['results']['selected']['correct']
            noctx=report['results']['no_context']['correct']
            control=report['results']['trained_no_context']['correct']
            flags=dict(both_models_correct=selected and control,
                context_helps_same_weights=selected and not noctx,
                context_hurts_same_weights=noctx and not selected,
                two_stage_wins_control=selected and not control,
                control_wins_two_stage=control and not selected,
                two_stage_wrong=not selected)
            for category,found in flags.items():
                if found and category not in cases:
                    cases[category]=position
                    write_report(args.output/(category+'.json'),report)
            rows.append(dict(position=position,target=query['target'],original_size=query['original_size'],
                predictions={k:v['prediction'] for k,v in report['results'].items()},
                context_indices={k:v['context_index'] for k,v in report['results'].items()}))
            if position%25==0 or position+1==args.limit:
                print(f'CALTECH DEMO {position+1}/{args.limit} | cases={cases}',flush=True)
        after=[p._version for model in (engine.model,engine.control) for p in model.parameters()]
        if before!=after: raise ValueError('Demo mutated model parameters')
        summary=dict(status='complete',dataset='caltech101',queries=args.limit,test_evaluated=False,
            weights_updated=False,correct=counts,accuracy={k:v/args.limit for k,v in counts.items()},
            metadata=engine.metadata,cases=cases,rows=rows,
            warning='Batch-1 local validation demo replay, not a test benchmark or a new training run; '
                    'FP16/device/batch can differ slightly from cloud training validation.')
        write_report(args.output/'summary.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','metadata')},ensure_ascii=False,indent=2))
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        parser.exit(1,f'CALTECH DEMO VERIFY FAILED: {exc}\nKeep partial output; no completion claim.\n')


if __name__=='__main__': main()
