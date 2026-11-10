"""One disposable optimizer step on real CIFAR-100; not an accuracy experiment."""
import argparse
import json
from pathlib import Path
import time

import torch
from torchvision.datasets import CIFAR100

from tacs.data import cifar_transforms
from tacs.device import seed_everything
from tacs.effective_batch import backward_effective_window
from tacs.inference import default_artifacts, load_verified
from tacs.losses import TACSLoss
from tacs.metrics import PeakMemoryMonitor
from tacs.train import _autocast_context, _sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; refusing overwrite')
    settings = default_artifacts()
    torch.set_num_threads(2)
    seed_everything(44)
    if args.device == 'cuda':
        torch.set_float32_matmul_precision('high')
    loaded = load_verified(settings['checkpoint'], settings['standard_encoder'], settings['split_file'], args.device)
    device, model = loaded.device, loaded.model.to(loaded.device)
    amp = device.type == 'cuda'
    train_transform, evaluation = cifar_transforms('cifar100', 256, 'dinov3')
    data = CIFAR100(str(settings['data_root']), train=True, transform=evaluation, download=False)
    from PIL import Image
    query_ids = loaded.split.query_train[:64]
    pool_ids = loaded.split.candidate_pool[:32]
    candidate_images = torch.stack([data[i][0] for i in pool_ids])
    batches = []
    for offset in range(0, 64, 4):
        ids = query_ids[offset:offset+4]
        batches.append((torch.stack([train_transform(Image.fromarray(data.data[i])) for i in ids]),
                        torch.tensor([data.targets[i] for i in ids])))
    model.eval()
    with torch.no_grad(), _autocast_context(device, amp):
        bank = torch.cat([model.selector.encoder(b.to(device)).detach()
                          for b in candidate_images.split(4)])
    model.train()
    def forward(batch):
        query, target = batch[0].to(device), batch[1].to(device)
        selection = model.selector.forward_embedding_bank(query, bank)
        chosen = candidate_images[selection[1].detach().argmax(-1).cpu()].to(device)
        return model.forward_embedding_bank(query, bank, selected_context=chosen, selection=selection), target
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with PeakMemoryMonitor(device) as memory:
        stats = backward_effective_window(model, batches, forward, TACSLoss(1.),
                                          lambda: _autocast_context(device, amp), scaler)
        if amp:
            scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('Missing/non-finite gradients; step refused')
        norm = sum(float(g.float().square().sum()) for g in gradients) ** .5
        if norm == 0:
            raise RuntimeError('All gradients are zero')
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    assert _sha256_file(settings['checkpoint']) == loaded.metadata['checkpoint_sha256']
    result = dict(status='passed', kind='technical_real_data_smoke_not_accuracy',
                  samples=64, micro_batch=4, accumulation=16, candidate_subset=32,
                  optimizer_steps=1, in_memory_weights_updated=True, gradient_l2=norm, statistics=stats,
                  memory=memory.result().as_dict(), elapsed_s=time.perf_counter()-started,
                  provenance=loaded.metadata, source_checkpoint_unchanged=True,
                  test_evaluated=False,
                  warning='Existing weights used only in memory, no trained checkpoint saved. Candidate subset is NOT the full Two-Stage benchmark. No accuracy improvement claimed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
