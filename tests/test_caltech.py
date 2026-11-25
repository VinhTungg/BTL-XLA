"""Unit fixtures, not substitute Caltech data or reported model accuracy."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from tacs.caltech.data import (
    CaltechImages, atomic_json, datasets, digest, make_split, prepare_train_subset, scan, validate_split,
)
from tacs.caltech.cosine import CosineTop1Model, PairedClassificationLoss
from tacs.caltech.run import AnchoredConfig, Config, fit, require_manifest, restore
from tacs.baselines import BaselineModel
from tacs.losses import TACSLoss
from tacs.model import TACSOutput, TinyVisionEncoder, TinyPatchTokenTaskNetwork
from tacs.train import evaluate


class CaltechTest(unittest.TestCase):
    def inventory(self, classes=3, count=34):
        return dict(classes=[f'class{i}' for i in range(classes)], inventory_sha256='unit-only',
            items=[dict(file=f'c{c}/{i}.jpg',target=c,sha256=f'{c}-{i}',width=300,height=200)
                   for c in range(classes) for i in range(count)])

    def test_split_is_reproducible_balanced_and_disjoint(self):
        inv = self.inventory()
        a, b = make_split(inv), make_split(inv)
        self.assertEqual(a, b)
        self.assertEqual([len(a[k]) for k in ('query_train','candidate_pool','validation','test')], [57,18,15,12])
        self.assertNotEqual(a['query_train'], make_split(inv, 43)['query_train'])

    def test_training_subset_is_balanced_persistent_and_disjoint(self):
        inv = self.inventory()
        split = make_split(inv)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root/'split.json', split)
            self.assertEqual(prepare_train_subset(root, inv, split, None), (split['query_train'], {}))
            indices, identity = prepare_train_subset(root, inv, split, 5)
            self.assertEqual(Counter(inv['items'][i]['target'] for i in indices), {0: 5, 1: 5, 2: 5})
            self.assertLess(set(indices), set(split['query_train']))
            self.assertEqual(identity['train_subset_sha256'], digest(root/'train_subset.json'))
            self.assertEqual(prepare_train_subset(root, inv, split, 5), (indices, identity))
            train, pool, val = datasets(root, inv, split, train_indices=indices)
            self.assertEqual(train.indices, indices)
            self.assertEqual(pool.indices, split['candidate_pool'])
            self.assertEqual(val.indices, split['validation'])
            with self.assertRaisesRegex(ValueError, 'subset differs'):
                prepare_train_subset(root, inv, split, 4)
            with self.assertRaisesRegex(ValueError, 'supply its'):
                prepare_train_subset(root, inv, split, None)
            for count in (0, 20):
                with self.assertRaisesRegex(ValueError, 'between 1 and 19'):
                    prepare_train_subset(root, inv, split, count)

    def test_overlap_and_wrong_inventory_refused(self):
        inv = self.inventory()
        for mode in ('overlap','inventory','classmap'):
            s = make_split(inv)
            if mode=='overlap': s['candidate_pool'][0] = s['validation'][0]
            if mode=='inventory': s['inventory_sha256'] = 'wrong'
            if mode=='classmap': s['classes'] = ['wrong']
            with self.assertRaises(ValueError): validate_split(s, inv)

    def test_duplicate_content_crossing_roles_refused(self):
        inv = self.inventory()
        s = make_split(inv)
        inv['items'][s['test'][0]]['sha256'] = inv['items'][s['candidate_pool'][0]]['sha256']
        with self.assertRaisesRegex(ValueError, 'Byte-identical'):
            validate_split(s, inv)

    def test_grayscale_rgb_and_background_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'caltech101/101_ObjectCategories'
            (root/'object').mkdir(parents=True)
            (root/'BACKGROUND_Google').mkdir()
            for i in range(31):
                Image.new('L', (40,30), i).save(root/'object'/f'image_{i:04d}.jpg')
            inv = scan(temp, expected_classes=1)
            self.assertEqual(inv['classes'], ['object'])
            self.assertEqual(len(inv['items']), 31)
            image, target = CaltechImages(temp, inv)[0]
            self.assertEqual((image.mode, image.size, target), ('RGB',(40,30),0))
            with self.assertRaises(ValueError): scan(temp)

    def test_manifest_refuses_changed_recipe_and_unowned_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run'
            require_manifest(path, {'schema':1})
            require_manifest(path, {'schema':1})
            with self.assertRaises(ValueError): require_manifest(path, {'schema':2})
            other = Path(temp)/'unowned'
            other.mkdir()
            (other/'user.txt').write_text('keep')
            with self.assertRaises(ValueError): require_manifest(other, {'schema':1})
            self.assertEqual((other/'user.txt').read_text(), 'keep')

    def test_config_rejects_wrong_dataset_head_and_schedule(self):
        for kw in [dict(dataset='cifar100'),dict(num_classes=100),dict(epochs=5),dict(batch_size=0),
                   dict(selector_temperature=0),dict(policy_weight=float('nan')),dict(topk_candidates=0)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                Config(baseline='two_stage', **kw)

    def test_full_training_defaults_to_100_epochs(self):
        for name in ('no_context','two_stage'):
            self.assertEqual(Config(baseline=name).epochs,100)

    def test_anchored_config_preserves_the_original_checkpoint_recipe(self):
        original = asdict(Config(baseline='two_stage'))
        anchored = asdict(AnchoredConfig(baseline='two_stage'))
        self.assertEqual(anchored.pop('aux_query_weight'), 0.5)
        self.assertEqual(anchored, original)
        with self.assertRaises(ValueError):
            AnchoredConfig(baseline='no_context')

    def test_validation_counts_context_help_and_harm_on_same_weights(self):
        class Model(torch.nn.Module):
            def forward(self, query, candidates):
                logits = torch.tensor([[2., 0.], [0., 2.]])
                return TACSOutput(
                    context_logits=logits[torch.tensor([0, 1, 0, 0])[query]],
                    no_context_logits=logits[torch.tensor([0, 0, 1, 0])[query]],
                    selector_scores=torch.zeros(len(query), 1),
                    selection_weights=torch.ones(len(query), 1),
                    selector_temperature=0.1,
                )

        loader = DataLoader(TensorDataset(torch.arange(4), torch.tensor([0, 1, 1, 1])), batch_size=2)
        result = evaluate(Model(), loader, TACSLoss(), torch.device('cpu'), compare_no_context=True)
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['no_context_accuracy'], 0.5)
        self.assertEqual(result['context_helped'], 1)
        self.assertEqual(result['context_hurt'], 1)

    def test_fixed_cosine_arm_trains_and_records_paired_validation(self):
        cfg = AnchoredConfig(baseline='cosine_top1', epochs=1, warmup_epochs=0,
                             policy_weight=0, topk_candidates=1, gradient_accumulation_steps=1)
        with self.assertRaises(ValueError):
            AnchoredConfig(baseline='cosine_top1')
        model = CosineTop1Model(TinyPatchTokenTaskNetwork(101, 8))
        query = torch.rand(4, 3, 16, 16)
        context = torch.rand(4, 1, 3, 16, 16)
        targets = torch.arange(4)
        output = model(query, context)
        loss = PairedClassificationLoss(cfg.aux_query_weight)(output, targets)
        grads = torch.autograd.grad(loss, (output.context_logits, output.no_context_logits))
        self.assertTrue(all(grad.abs().sum().item() > 0 for grad in grads))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            model(query, context.repeat(1, 2, 1, 1, 1))

        rows = [dict(query=q, candidates=c, target=y) for q, c, y in zip(query, context, targets)]
        loader = DataLoader(rows, batch_size=4)
        inventory = dict(classes=[str(i) for i in range(101)])
        with tempfile.TemporaryDirectory() as temp:
            result = fit(cfg, model, loader, loader, temp, inventory, torch.device('cpu'))
            self.assertEqual(result['status'], 'complete')
            metrics = json.loads((Path(temp)/'metrics.json').read_text(encoding='utf-8'))
            epoch = metrics['history'][0]
            self.assertAlmostEqual(
                epoch['val_acc'] - epoch['val_no_context_acc'],
                (epoch['val_context_helped'] - epoch['val_context_hurt']) / len(rows),
            )

    def test_real_epoch_engine_resume_roundtrip_on_unit_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            torch.set_num_threads(2)
            cfg = Config(baseline='no_context', epochs=2, warmup_epochs=0,
                         gradient_accumulation_steps=1)
            inventory = dict(classes=[str(i) for i in range(101)], inventory_sha256='unit')
            loader = DataLoader(TensorDataset(torch.rand(8,3,32,32),torch.arange(8)),batch_size=4)
            def model():
                return BaselineModel(101,8,'no_context',TinyVisionEncoder(8),TinyPatchTokenTaskNetwork(101,8))
            output = Path(temp)
            result = fit(cfg,model(),loader,loader,output,inventory,torch.device('cpu'),stop_after=1)
            self.assertEqual(result['status'],'partial')
            self.assertFalse((output/'result.json').exists())
            first = torch.load(output/'checkpoint_latest.pt',weights_only=True)
            self.assertEqual(first['epoch'],1)
            with self.assertRaisesRegex(ValueError,'config/history mismatch'):
                # Reject changed total schedule before touching model/optimizer.
                restore(output/'checkpoint_latest.pt',None,None,None,None,
                        Config(baseline='no_context',epochs=100))
            with self.assertRaisesRegex(ValueError, 'training subset mismatch'):
                restore(output/'checkpoint_latest.pt', None, None, None, None, cfg,
                        train_subset_sha256='other-training-subset')
            result = fit(cfg,model(),loader,loader,output,inventory,torch.device('cpu'))
            self.assertEqual(result['status'],'complete')
            latest = torch.load(output/'checkpoint_latest.pt',weights_only=True)
            self.assertEqual([h['epoch'] for h in latest['history']],[1,2])
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(result['reference_encoder_sha256'],digest(output/'best_encoder.pt'))
            self.assertEqual(latest['config'], asdict(cfg))


if __name__ == '__main__':
    unittest.main()
