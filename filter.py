import os
import json
import argparse
import logging
from tqdm import tqdm
import clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    tokenize = clip.tokenize

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    kept = []
    dropped_invalid = 0
    dropped_too_long = 0

    for d in tqdm(data, desc="Filtering"):
        try:
            if d.get("status", "") != "success":
                dropped_invalid += 1
                continue

            instruction = d.get("instruction", "").strip()
            target_cap = d.get("target_caption", "").strip()

            if not instruction or not target_cap:
                dropped_invalid += 1
                continue

            inst_len = int((tokenize(instruction, truncate=False)[0] != 0).sum().item())
            tgt_len = int((tokenize(target_cap, truncate=False)[0] != 0).sum().item())

            if inst_len > 77 or tgt_len > 77:
                dropped_too_long += 1
                continue

            kept.append(d)

        except Exception as e:
            logging.warning(f"Failed sample {d.get('id', 'unknown')}: {e}")
            dropped_invalid += 1

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    logging.info(f"Total: {len(data)}")
    logging.info(f"Kept: {len(kept)}")
    logging.info(f"Dropped invalid: {dropped_invalid}")
    logging.info(f"Dropped too long: {dropped_too_long}")
    logging.info(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    main()