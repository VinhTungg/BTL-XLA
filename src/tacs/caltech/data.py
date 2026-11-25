"""Real Caltech-101 RGB data and a persistent, disjoint demo split."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset, Subset

PROTOCOL = 'caltech101_demo_19query_6pool_5val_rest_test_v1'
TRAIN_PER_CLASS = 19


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def scan(root, expected_classes=101):
    root = Path(root)/'caltech101/101_ObjectCategories'
    if not root.is_dir():
        raise FileNotFoundError(f'Caltech-101 not found: {root}; run preparation with download enabled')
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != 'BACKGROUND_Google')
    if len(classes) != expected_classes:
        raise ValueError(f'Expected {expected_classes} object classes excluding BACKGROUND_Google, got {len(classes)}')
    items = []
    for target, name in enumerate(classes):
        files = sorted((root/name).glob('*.jpg'))
        if len(files) < 31:
            raise ValueError(f'{name}: need at least 31 images for disjoint demo roles, got {len(files)}')
        for path in files:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            items.append(dict(file=path.relative_to(root).as_posix(), target=target,
                              sha256=digest(path), width=width, height=height))
    fingerprint = hashlib.sha256(json.dumps(dict(classes=classes, items=items),
                                 sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(classes=classes, items=items, inventory_sha256=fingerprint)


def make_split(inventory, seed=42):
    rng = random.Random(seed)
    split = dict(protocol=PROTOCOL, seed=seed, inventory_sha256=inventory['inventory_sha256'],
                 classes=inventory['classes'], query_train=[], candidate_pool=[], validation=[], test=[])
    for target in range(len(inventory['classes'])):
        indices = [i for i, row in enumerate(inventory['items']) if row['target'] == target]
        rng.shuffle(indices)
        split['query_train'].extend(indices[:TRAIN_PER_CLASS])
        split['candidate_pool'].extend(indices[TRAIN_PER_CLASS:25])
        split['validation'].extend(indices[25:30])
        split['test'].extend(indices[30:])
    validate_split(split, inventory)
    return split


def validate_split(split, inventory):
    if (split.get('protocol') != PROTOCOL or split.get('classes') != inventory['classes']
            or split.get('inventory_sha256') != inventory['inventory_sha256']):
        raise ValueError('Dataset inventory/classes/protocol mismatch')
    seen, content_roles = set(), {}
    for role in ('query_train', 'candidate_pool', 'validation', 'test'):
        indices = split[role]
        if not indices or any(type(i) is not int or not 0 <= i < len(inventory['items']) for i in indices):
            raise ValueError(f'Invalid/empty {role} indices')
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError('Duplicate/overlapping split roles')
        seen.update(indices)
        counts = Counter(inventory['items'][i]['target'] for i in indices)
        required = dict(query_train=TRAIN_PER_CLASS, candidate_pool=6, validation=5).get(role)
        if any((counts[t] != required if required is not None else counts[t] < 1)
               for t in range(len(inventory['classes']))):
            raise ValueError(f'Wrong class balance for {role}')
        for i in indices:
            sha = inventory['items'][i]['sha256']
            previous = content_roles.setdefault(sha, role)
            if previous != role:
                raise ValueError('Byte-identical image crosses split roles; do not train until split is reviewed')
    if seen != set(range(len(inventory['items']))):
        raise ValueError('Split does not cover inventory exactly once')


def prepare(data_root, split_file, download=False):
    if download and not (Path(data_root)/'caltech101/101_ObjectCategories').is_dir():
        from torchvision.datasets import Caltech101
        Caltech101(str(data_root), target_type='category', download=True)
    inventory = scan(data_root)
    path = Path(split_file)
    if path.is_file():
        split = json.loads(path.read_text(encoding='utf-8'))
        validate_split(split, inventory)
    else:
        split = make_split(inventory)
        atomic_json(path, split)
    return inventory, split


def prepare_train_subset(output_root, inventory, split, train_per_class):
    """Select a fixed prefix per class from the already shuffled training split."""
    subset_path = Path(output_root)/'train_subset.json'
    if train_per_class is None:
        if subset_path.exists():
            raise ValueError('This output root has a training subset; supply its --train-per-class')
        return split['query_train'], {}
    if type(train_per_class) is not int or not 1 <= train_per_class <= TRAIN_PER_CLASS:
        raise ValueError(f'train-per-class must be between 1 and {TRAIN_PER_CLASS}')

    counts = Counter()
    indices = []
    for idx in split['query_train']:
        target = inventory['items'][idx]['target']
        if counts[target] < train_per_class:
            counts[target] += 1
            indices.append(idx)
    if len(counts) != len(inventory['classes']) or set(counts.values()) != {train_per_class}:
        raise ValueError('Cannot select the requested number of training images for every class')
    subset = dict(train_per_class=train_per_class, selection='prefix_per_class_in_split_order',
                  split_sha256=digest(subset_path.parent/'split.json'), query_train=indices)
    if subset_path.is_file():
        if json.loads(subset_path.read_text(encoding='utf-8')) != subset:
            raise ValueError('Training subset differs from this output root; use a new output root')
    else:
        if any(subset_path.parent.glob('*/experiment_manifest.json')):
            raise ValueError('Training runs already exist here; use a new output root for a new subset')
        atomic_json(subset_path, subset)
    return indices, dict(train_per_class=train_per_class, train_subset_sha256=digest(subset_path))


class CaltechImages(Dataset):
    def __init__(self, data_root, inventory, transform=None):
        self.root = Path(data_root)/'caltech101/101_ObjectCategories'
        self.inventory, self.transform = inventory, transform
        self.classes = inventory['classes']
        self.targets = [row['target'] for row in inventory['items']]

    def __len__(self):
        return len(self.targets)

    def raw_image(self, index):
        path = self.root/self.inventory['items'][index]['file']
        with Image.open(path) as source:
            return source.convert('RGB')

    def __getitem__(self, index):
        image = self.raw_image(index)
        return (self.transform(image) if self.transform else image), self.targets[index]


def transforms(train=False):
    from torchvision import transforms as T
    spatial = [T.RandomResizedCrop(256), T.RandomHorizontalFlip()] if train else [T.Resize((256, 256))]
    # No vertical flip for this demo protocol; do not call it the paper's full recipe.
    return T.Compose([*spatial, T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])


def datasets(data_root, inventory, split, *, train_indices=None):
    training = CaltechImages(data_root, inventory, transforms(True))
    evaluation = CaltechImages(data_root, inventory, transforms(False))
    return (Subset(training, split['query_train'] if train_indices is None else train_indices),
            Subset(evaluation, split['candidate_pool']),
            Subset(evaluation, split['validation']))
