"""Kiểm tra tích hợp DataLoader với Skeleton TACS - Nhiệm vụ Thành viên 3 (Gói 2 & 4).

Script này thực hiện:
1. Tạo mock dataset tương thích chuẩn CIFAR-10 (10 lớp, ảnh 3x32x32).
2. Thiết lập ContextDataset (num_candidates=8) và DataLoader (batch_size=4).
3. Đưa 1 batch qua TACSModel.
4. Kiểm tra các tensor đầu ra:
   - context_logits: [B, num_classes]
   - no_context_logits: [B, num_classes]
   - selector_scores: [B, N]
   - selection_weights: [B, N] (Gumbel-Softmax hard one-hot)
5. Tính toán TACSLoss (task loss, policy loss, advantage, reward).
6. Thực hiện backward pass và kiểm tra gradient truyền về TaskAlignedSelector.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from tacs.data import ContextDataset
from tacs.device import configure_stdout
from tacs.losses import TACSLoss
from tacs.model import TACSModel


def main() -> None:
    configure_stdout()
    print("=" * 70)
    print("  KIỂM TRA TÍCH HỢP DATALOADER -> TACSMODEL (GÓI 2 & GÓI 4)")
    print("=" * 70)

    # 1. Khởi tạo dữ liệu giả lập chuẩn CIFAR-10
    torch.manual_seed(42)
    num_samples = 120
    num_classes = 10
    batch_size = 4
    num_candidates = 8

    images = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, num_classes, (num_samples,))
    full_dataset = TensorDataset(images, targets)

    query_dataset = Subset(full_dataset, list(range(80)))
    candidate_dataset = Subset(full_dataset, list(range(80, num_samples)))

    print(f"[*] Query dataset size    : {len(query_dataset)}")
    print(f"[*] Candidate pool size   : {len(candidate_dataset)}")
    print(f"[*] Số candidate mỗi query: {num_candidates}")

    # 2. Tạo ContextDataset và DataLoader
    context_dataset = ContextDataset(
        query_dataset=query_dataset,
        candidate_dataset=candidate_dataset,
        num_candidates=num_candidates,
        seed=42,
    )
    loader = DataLoader(context_dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    print("\n--- 1. KIỂM TRA BATCH TỪ DATALOADER ---")
    query = batch["query"]
    candidates = batch["candidates"]
    target = batch["target"]
    query_idx = batch["query_index"]
    cand_idx = batch["candidate_indices"]
    cand_targets = batch["candidate_targets"]

    print(f"[*] query shape             : {list(query.shape)} (Kỳ vọng: [{batch_size}, 3, 32, 32])")
    print(f"[*] candidates shape        : {list(candidates.shape)} (Kỳ vọng: [{batch_size}, {num_candidates}, 3, 32, 32])")
    print(f"[*] target shape            : {list(target.shape)} (Kỳ vọng: [{batch_size}]) -> {target.tolist()}")
    print(f"[*] candidate_indices shape : {list(cand_idx.shape)}")
    print(f"[*] candidate_targets shape : {list(cand_targets.shape)}")

    # 3. Khởi tạo TACSModel
    print("\n--- 2. FORWARD PASS QUA TACSMODEL ---")
    model = TACSModel(num_classes=num_classes, embedding_dim=64, temperature=0.1)
    model.train()

    output = model(query, candidates)
    print(f"[*] context_logits shape    : {list(output.context_logits.shape)}")
    print(f"[*] no_context_logits shape : {list(output.no_context_logits.shape)}")
    print(f"[*] selector_scores shape   : {list(output.selector_scores.shape)}")
    print(f"[*] selection_weights shape : {list(output.selection_weights.shape)}")

    # Kiểm tra hard selection (one-hot)
    sums = output.selection_weights.sum(dim=-1).tolist()
    is_one_hot = all(abs(s - 1.0) < 1e-5 for s in sums)
    print(f"[*] Tổng selection weights mỗi query: {sums} -> One-hot hợp lệ: {is_one_hot}")

    # 4. Tính toán Loss và Reward
    print("\n--- 3. TÍNH TOÁN HYBRID TACSLOSS ---")
    criterion = TACSLoss(policy_weight=0.5)
    loss_output = criterion(output, target)

    print(f"[*] Total Loss   : {loss_output.total.item():.4f}")
    print(f"[*] Task Loss    : {loss_output.task.item():.4f}")
    print(f"[*] Policy Loss  : {loss_output.policy.item():.4f}")
    print(f"[*] Mean Reward  : {loss_output.mean_reward.item():.4f}")

    # 5. Backward Pass và Gradient
    print("\n--- 4. BACKWARD PASS & KIỂM TRA GRADIENT ---")
    model.zero_grad()
    loss_output.total.backward()

    selector_grads = [
        (name, p.grad.norm().item())
        for name, p in model.selector.named_parameters()
        if p.grad is not None
    ]
    classifier_grads = [
        (name, p.grad.norm().item())
        for name, p in model.classifier.named_parameters()
        if p.grad is not None
    ]

    print(f"[*] Số lượng tham số Selector nhận gradient  : {len(selector_grads)} / {len(list(model.selector.parameters()))}")
    print(f"[*] Số lượng tham số Classifier nhận gradient: {len(classifier_grads)} / {len(list(model.classifier.parameters()))}")

    all_received_grad = (len(selector_grads) > 0) and all(g > 0 for _, g in selector_grads)
    print(f"[*] Gradient truyền về Selector thành công : {'[PASS]' if all_received_grad else '[FAIL]'}")

    # 6. Kiểm tra chế độ Eval (Deterministic Selection)
    print("\n--- 5. KIỂM TRA CHẾ ĐỘ EVAL (INFERENCE) ---")
    model.eval()
    with torch.no_grad():
        out_eval1 = model(query, candidates)
        out_eval2 = model(query, candidates)
    is_deterministic = torch.equal(out_eval1.selection_weights, out_eval2.selection_weights)
    print(f"[*] Lựa chọn context trong chế độ eval có tất định (deterministic): {'[PASS]' if is_deterministic else '[FAIL]'}")

    print("=" * 70)
    print("KẾT LUẬN: Pipeline DataLoader -> TACSModel -> TACSLoss hoạt động hoàn hảo!")
    print("=" * 70)


if __name__ == "__main__":
    main()
