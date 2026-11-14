"""Real-checkpoint inference rehearsal. Never train or open CIFAR test data."""
import argparse
import gc
import json
from pathlib import Path

import torch
from tacs.showcase.__main__ import PRESETS, create_engine, make_report, validation_input, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        summary = []
        for preset in PRESETS:
            engine = create_engine(preset)
            for position in (0, 120, 111):
                image, query = validation_input(engine, position)
                report = make_report(engine, image, query)
                if (report['metadata']['checkpoint_sha256'] != PRESETS[preset][1]
                        or report['metadata']['weights_updated'] is not False
                        or report['metadata']['test_evaluated'] is not False):
                    raise ValueError('Unexpected checkpoint/provenance')
                if any(not 0 <= item['top5'][0]['probability'] <= 1 for item in report['results'].values()):
                    raise ValueError('Invalid probability')
                write_report(args.output/f'{preset}_{position}.json', report)
                item = dict(preset=preset, position=position, target=query['target'],
                            predictions={k:v['prediction'] for k,v in report['results'].items()})
                summary.append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
            del engine, report
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_report(args.output/'summary.json', dict(status='complete', cases=summary,
                     test_evaluated=False, weights_updated=False,
                     warning='Selected demonstration cases, not an accuracy benchmark.'))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'REHEARSAL FAILED: {exc}\nKeep partial outputs; do not treat as complete.\n')


if __name__ == '__main__':
    main()
