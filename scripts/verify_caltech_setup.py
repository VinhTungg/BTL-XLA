"""One real-data forward/backward; no optimizer step, checkpoint or accuracy claim."""
import argparse
import gc
import json
from pathlib import Path

import torch
from torch import nn

from tacs.caltech.data import datasets, prepare
from tacs.caltech.run import Config, build_model
from tacs.losses import TACSLoss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--split',type=Path,default=Path('artifacts/caltech_demo/split.json'))
    args=parser.parse_args()
    inventory,split=prepare(args.data_root,args.split)
    query,pool,_=datasets(args.data_root,inventory,split)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    q,y=query[0]
    images=torch.stack([pool[i][0] for i in range(2)]).to(device)
    q=q.unsqueeze(0).to(device)
    target=torch.tensor([y],device=device)
    results=[]
    for name in ('no_context','two_stage'):
        model=build_model(Config(baseline=name),device)
        model.train()
        if name=='no_context':
            logits=model(q)
            loss=nn.CrossEntropyLoss()(logits,target)
        else:
            # Production embedding-bank route; two real pool images only for this smoke.
            with torch.no_grad():
                model.selector.encoder.eval()
                bank=model.selector.encoder(images).detach()
            model.train()
            selection=model.selector.forward_embedding_bank(q,bank)
            context=images[selection[1].detach().argmax(dim=-1)]
            output=model.forward_embedding_bank(q,bank,selected_context=context,selection=selection)
            logits=output.context_logits
            loss=TACSLoss()(output,target).total
        if logits.shape!=(1,101) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise RuntimeError('Invalid logits/loss')
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Invalid gradients')
        results.append(dict(baseline=name,logits=list(logits.shape),finite_gradients=True,
                            loss=float(loss.detach()),optimizer_steps=0))
        print(name,'REAL-DATA FORWARD/BACKWARD PASS',flush=True)
        del model,logits,loss,grads
        if name=='two_stage':
            del bank,selection,context,output
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    print(json.dumps(dict(passed=True,device=str(device),classes=len(inventory['classes']),
        images=len(inventory['items']),test_evaluated=False,runs=results,
        note='Technical smoke only; no optimizer step or training checkpoint; no accuracy claim.'),indent=2))


if __name__=='__main__':
    main()
