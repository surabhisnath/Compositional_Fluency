"""Query GPT-4.1 true/false for each (food, feature) pair and cache to a pk.

Foods = unique lundin food responses (keys of files/lundin_freq_abs_log.json).
Features = files/food_features.txt (each stored as 'feature_<description>').
Output = files/features_gpt41_foods.pk  (SEPARATE from the animal features_gpt41.pk;
never overwritten). Resumable: reruns skip (food, feature) pairs already cached.
"""

import argparse
import json
import os
import pickle as pk
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

BASE = "../../"
MODEL = "gpt-4.1"
OUT = BASE + "files/features_gpt41_foods.pk"
SYSTEM = "You are a helpful assistant and food expert who has access to all the facts about foods."


def load_key():
    """Use the first OPENAI_API_KEY in keys.json that authenticates."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            OpenAI().models.list()
            return None  # env key works; let OpenAI() pick it up
        except Exception:
            pass
    raw = open("keys.json").read()
    for m in re.finditer(r'"OPENAI_API_KEY"\s*:\s*"([^"]+)"', raw):
        try:
            OpenAI(api_key=m.group(1)).models.list()
            return m.group(1)
        except Exception:
            continue
    raise SystemExit("No working OPENAI_API_KEY (env or keys.json)")


def judge(client, food, desc, retries=4):
    """Return 'true'/'false' (lowercased model output) for `food: desc`."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Output only true or false. {food}: {desc}"},
    ]
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=5, temperature=0
            )
            return r.choices[0].message.content.strip().lower()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-foods", type=int, default=None, help="test on first N foods")
    args = ap.parse_args()

    key = load_key()
    client = OpenAI(api_key=key) if key else OpenAI()

    descs = [l.strip() for l in open(BASE + "files/food_features.txt") if l.strip()]
    features = [(f"feature_{d}", d) for d in descs]  # (key, description)

    foods = sorted(json.load(open(BASE + "files/lundin_freq_abs_log.json")))
    if args.limit_foods:
        foods = foods[: args.limit_foods]
    print(f"{len(foods)} foods x {len(features)} features = {len(foods) * len(features)} pairs")

    try:
        features_dict = pk.load(open(args.out, "rb"))
    except Exception:
        features_dict = {}
    done0 = sum(len(v) for v in features_dict.values())
    print(f"resuming: {len(features_dict)} foods, {done0} pairs already cached")

    for fi, food in enumerate(foods):
        features_dict.setdefault(food, {})
        todo = [(k, d) for k, d in features if k not in features_dict[food]]
        if not todo:
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge, client, food, d): k for k, d in todo}
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    features_dict[food][k] = fut.result()
                except Exception as e:
                    print(f"  ERROR {food} / {k}: {e}")
        pk.dump(features_dict, open(args.out, "wb"))  # checkpoint per food
        pairs = sum(len(v) for v in features_dict.values())
        print(f"[{fi + 1}/{len(foods)}] {food}: {len(features_dict[food])}/{len(features)} feats "
              f"| total pairs {pairs}", flush=True)

    print(f"DONE. {len(features_dict)} foods -> {args.out}")


if __name__ == "__main__":
    main()
