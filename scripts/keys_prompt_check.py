#!/usr/bin/env python
"""Does the trained policy listen to the note text? An offline check, no robot.

For start frames of recorded episodes it asks the policy for its next action chunk six times, once per note
text, on the very same picture and arm state. If the text steers it, the planned motion differs between
texts and the text that was really used gives the smallest error against the recorded motion.

    uv run scripts/keys_prompt_check.py <policy folder> <dataset folder>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

JOINT = 0  # shoulder_pan: the joint that moves the fingertip from key to key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", type=Path, help=".../checkpoints/005000/pretrained_model")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--per-task", type=int, default=3, help="Episodes per note")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 45, 90], help="Frames inside each episode")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args(argv)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(args.policy)
    config.pretrained_path = args.policy
    config.device = args.device
    policy = get_policy_class(config.type).from_pretrained(args.policy, config=config)
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.policy),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    dataset = LeRobotDataset(f"local/{args.dataset.name}", root=args.dataset)
    episodes = dataset.meta.episodes
    tasks = list(dict.fromkeys(str(list(t)[0]) for t in episodes["tasks"]))
    tasks.sort(key=lambda t: ["do", "re", "mi", "fa", "sol", "la"].index(t.split()[1].strip(",")))
    chunk = config.chunk_size
    rows = []
    rng = np.random.default_rng(0)
    for true_index, task in enumerate(tasks):
        mine = [i for i, t in enumerate(episodes["tasks"]) if str(list(t)[0]) == task]
        for ep in rng.choice(mine, size=min(args.per_task, len(mine)), replace=False):
            start, end = int(episodes["dataset_from_index"][ep]), int(episodes["dataset_to_index"][ep])
            for frame in args.frames:
                if start + frame + chunk > end:
                    continue
                item = dataset[start + frame]
                truth = torch.stack([dataset[start + frame + k]["action"] for k in range(0, chunk, 7)]).numpy()
                plans = []
                noise = torch.randn(1, chunk, config.max_action_dim, generator=torch.Generator().manual_seed(0))
                for prompt in tasks:
                    obs = {k: v for k, v in item.items() if k.startswith("observation.")}
                    obs["task"] = prompt
                    batch = preprocessor(obs)
                    with torch.no_grad():
                        policy.reset()
                        out = policy.predict_action_chunk(batch, noise=noise.to(args.device))
                    plan = postprocessor(out).squeeze(0).float().cpu().numpy()
                    plans.append(plan[::7])
                plans_arr = np.stack(plans)  # (prompts, steps, joints)
                errors = np.abs(plans_arr[:, :, JOINT] - truth[None, :, JOINT]).mean(axis=1)
                rows.append(
                    {
                        "true": true_index,
                        "frame": frame,
                        "best": int(np.argmin(errors)),
                        "err_true": float(errors[true_index]),
                        "err_other": float(np.delete(errors, true_index).mean()),
                        "spread": float(np.ptp(plans_arr[:, -1, JOINT])),
                        "end": plans_arr[:, -1, JOINT],
                        "truth_end": float(truth[-1, JOINT]),
                    }
                )

    print(f"\nPolicy: {args.policy}\nNotes: {[t.split()[1].strip(',') for t in tasks]}")
    print("Pan = shoulder_pan in degrees at the end of the planned chunk (about 1.6 s ahead).\n")
    for frame in args.frames:
        group = [r for r in rows if r["frame"] == frame]
        if not group:
            continue
        ends = np.stack([r["end"] for r in group]).mean(axis=0)
        truth = [np.mean([r["truth_end"] for r in group if r["true"] == k]) for k in range(len(tasks))]
        hit = np.mean([r["best"] == r["true"] for r in group])
        print(f"frame {frame:3d} ({frame / dataset.fps:.1f} s into the episode), {len(group)} samples")
        print(f"   recorded pan per note      : {np.round(truth, 1)}")
        print(f"   planned pan per note TEXT  : {np.round(ends, 1)}   (same pictures, text swapped)")
        print(f"   pan spread across the texts: {np.mean([r['spread'] for r in group]):.1f} deg")
        print(
            f"   error with the right text {np.mean([r['err_true'] for r in group]):.1f} deg, "
            f"with a wrong text {np.mean([r['err_other'] for r in group]):.1f} deg; "
            f"right text is the best fit in {hit:.0%} (chance {1 / len(tasks):.0%})\n"
        )


if __name__ == "__main__":
    main()
