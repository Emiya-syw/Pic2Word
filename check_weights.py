#!/usr/bin/env python3
import argparse
import sys
from typing import Dict, Any

import torch


def maybe_strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_ckpt(path: str):
    print(f"[INFO] Loading checkpoint: {path}")
    return torch.load(path, map_location="cpu")


def compare_tensor_dict(name: str, sd1: Dict[str, torch.Tensor], sd2: Dict[str, torch.Tensor]) -> bool:
    print(f"\n===== Comparing {name} =====")
    ok = True

    sd1 = maybe_strip_module_prefix(sd1)
    sd2 = maybe_strip_module_prefix(sd2)

    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    only1 = sorted(keys1 - keys2)
    only2 = sorted(keys2 - keys1)
    common = sorted(keys1 & keys2)

    if only1:
        ok = False
        print(f"[DIFF] Keys only in ckpt1 ({len(only1)}):")
        for k in only1:
            print(f"  - {k}")

    if only2:
        ok = False
        print(f"[DIFF] Keys only in ckpt2 ({len(only2)}):")
        for k in only2:
            print(f"  - {k}")

    equal_count = 0
    diff_count = 0

    for k in common:
        v1 = sd1[k]
        v2 = sd2[k]

        if not isinstance(v1, torch.Tensor) or not isinstance(v2, torch.Tensor):
            if v1 != v2:
                ok = False
                diff_count += 1
                print(f"[DIFF] {k}: non-tensor value differs")
                print(f"       ckpt1={v1}")
                print(f"       ckpt2={v2}")
            else:
                equal_count += 1
            continue

        if v1.shape != v2.shape:
            ok = False
            diff_count += 1
            print(f"[DIFF] {k}: shape differs")
            print(f"       ckpt1 shape={tuple(v1.shape)}")
            print(f"       ckpt2 shape={tuple(v2.shape)}")
            continue

        if v1.dtype != v2.dtype:
            ok = False
            diff_count += 1
            print(f"[DIFF] {k}: dtype differs")
            print(f"       ckpt1 dtype={v1.dtype}")
            print(f"       ckpt2 dtype={v2.dtype}")
            continue

        # “完全相同”比较：逐元素精确一致
        if torch.equal(v1, v2):
            equal_count += 1
        else:
            ok = False
            diff_count += 1
            max_abs = (v1.float() - v2.float()).abs().max().item()
            same_numel = int((v1 == v2).sum().item()) if v1.numel() > 0 else 0
            print(f"[DIFF] {k}: tensor values differ")
            print(f"       shape={tuple(v1.shape)}, dtype={v1.dtype}")
            print(f"       max_abs_diff={max_abs}")
            print(f"       equal_elements={same_numel}/{v1.numel()}")

    print(f"[SUMMARY] {name}: equal={equal_count}, diff={diff_count}, total_common={len(common)}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt1", required=True, help="Path to first checkpoint")
    parser.add_argument("--ckpt2", required=True, help="Path to second checkpoint")
    args = parser.parse_args()

    ckpt1 = load_ckpt(args.ckpt1)
    ckpt2 = load_ckpt(args.ckpt2)

    required_keys = ["state_dict", "state_dict_img2text"]

    for key in required_keys:
        if key not in ckpt1:
            print(f"[ERROR] {args.ckpt1} missing key: {key}")
            sys.exit(1)
        if key not in ckpt2:
            print(f"[ERROR] {args.ckpt2} missing key: {key}")
            sys.exit(1)

    ok_model = compare_tensor_dict("state_dict", ckpt1["state_dict"], ckpt2["state_dict"])
    ok_img2text = compare_tensor_dict("state_dict_img2text", ckpt1["state_dict_img2text"], ckpt2["state_dict_img2text"])

    print("\n===== FINAL RESULT =====")
    print(f"state_dict identical: {ok_model}")
    print(f"state_dict_img2text identical: {ok_img2text}")

    if ok_model and ok_img2text:
        print("[PASS] Checkpoints are exactly identical for both state_dict and state_dict_img2text.")
        sys.exit(0)
    else:
        print("[FAIL] Checkpoints differ.")
        sys.exit(2)


if __name__ == "__main__":
    main()