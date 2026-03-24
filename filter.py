import os
import json
import argparse
import logging
import re
from tqdm import tqdm
import clip


def contains_non_english_text(text: str) -> bool:
    """
    过滤明显非英语内容：
    - 中文
    - 日文
    - 韩文
    - 西里尔字母
    - 阿拉伯字母
    - 天城文等
    
    说明：
    1. 允许普通英文、数字、标点、空格
    2. 允许常见西欧重音字符可按需放开；这里默认严格一些，出现这些字符也可视情况保留
    3. 这是启发式规则，不是语言识别器
    """
    if not text:
        return True

    # 明显非英语文字范围
    non_english_pattern = re.compile(
        r"[\u4e00-\u9fff"      # CJK Unified Ideographs (中文)
        r"\u3400-\u4dbf"       # CJK Extension A
        r"\u3040-\u309f"       # Hiragana
        r"\u30a0-\u30ff"       # Katakana
        r"\uac00-\ud7af"       # Hangul
        r"\u0400-\u04ff"       # Cyrillic
        r"\u0600-\u06ff"       # Arabic
        r"\u0900-\u097f]"      # Devanagari
    )
    return bool(non_english_pattern.search(text))


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
    dropped_non_english = 0

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

            # 过滤非英语
            if contains_non_english_text(instruction) or contains_non_english_text(target_cap):
                dropped_non_english += 1
                continue

            inst_len = int((tokenize(instruction, truncate=False)[0] != 0).sum().item())
            tgt_len = int((tokenize(target_cap, truncate=False)[0] != 0).sum().item())

            if inst_len > 40 or tgt_len > 40:
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
    logging.info(f"Dropped non-english: {dropped_non_english}")
    logging.info(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    main()