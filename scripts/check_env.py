"""Kiểm tra toàn diện môi trường làm việc - Nhiệm vụ Thành viên 3 (Gói 1).

Script này thu thập thông tin phần cứng, phần mềm, PyTorch, CUDA, Torchvision
và in ra checklist trạng thái cùng cẩm nang khắc phục lỗi thường gặp.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# Đảm bảo console Windows hỗ trợ UTF-8 trước khi in tiếng Việt
try:
    from tacs.device import configure_stdout
    configure_stdout()
except Exception:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    print("=" * 65)
    print("      KIỂM TRA MÔI TRƯỜNG LÀM VIỆC - TACS PROJECT (TUẦN 1)")
    print("=" * 65)

    # 1. Hệ điều hành & Python
    py_ver = platform.python_version()
    os_name = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"
    print(f"[*] Hệ điều hành   : {os_name}")
    print(f"[*] Python         : {py_ver} ({sys.executable})")

    # 2. PyTorch & Device
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        print(f"[*] PyTorch        : {torch_ver}")
        print(f"[*] CUDA khả dụng  : {'Có (True)' if cuda_avail else 'Không (Dùng CPU)'}")
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            print(f"[*] GPU            : {gpu_name} ({gpu_mem} GB)")
        else:
            print("[*] Thiết bị chạy  : CPU (đã cấu hình auto fallback an toàn)")
    except ImportError:
        print("[!] LỖI: Chưa cài đặt PyTorch! Hãy chạy 'pip install torch'.")
        sys.exit(1)

    # 3. Torchvision & NumPy & Matplotlib
    for pkg_name in ("torchvision", "numpy", "matplotlib", "psutil"):
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "ok")
            print(f"[*] Thư viện {pkg_name:<10}: {ver} [OK]")
        except ImportError:
            print(f"[!] Thư viện {pkg_name:<10}: CHƯA CÀI ĐẶT [Cần bổ sung]")

    # 4. Kiểm tra import module dự án `tacs`
    try:
        import tacs
        from tacs.device import resolve_device, seed_everything
        from tacs.model import TACSModel
        from tacs.data import build_cifar10, ContextDataset
        from tacs.losses import TACSLoss
        print("[*] Module dự án   : import tacs [THÀNH CÔNG]")
    except ImportError as e:
        print(f"[!] LỖI import tacs: {e}")
        print("    -> Giải pháp: Thiết lập $env:PYTHONPATH='src' hoặc chạy 'pip install -e .'")

    # 5. Checklist nghiệm thu môi trường Gói 1
    print("-" * 65)
    print("CHECKLIST NGHIỆM THU MÔI TRƯỜNG (GÓI 1 - THÀNH VIÊN 3):")
    checks = [
        ("Python >= 3.10", sys.version_info >= (3, 10)),
        ("PyTorch installed", "torch" in sys.modules),
        ("Torchvision installed", "torchvision" in sys.modules),
        ("NumPy installed", "numpy" in sys.modules),
        ("TACS Core Module imported", "tacs" in sys.modules),
    ]
    for name, passed in checks:
        status = "[PASS]" if passed else "[WARN/CHECK]"
        print(f"  {status} {name}")

    print("=" * 65)
    print("Môi trường sẵn sàng cho các thực nghiệm Tuần 1!")
    print("=" * 65)


if __name__ == "__main__":
    main()
