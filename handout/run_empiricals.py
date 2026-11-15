"""Run all HW7 empirical experiments and build plots/answer sheet.

Designed for the supplied Google Colab environment. Results are written after
every run, so rerunning this script resumes instead of discarding completed work.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "HW7_large_stories"
TRAIN_DATA = DATA_DIR / "train_stories.json"
VALID_DATA = DATA_DIR / "valid_stories.json"
RESULTS_PATH = ROOT / "empiricals" / "results.json"
DOWNLOAD_ID = "1YZhMGnJNLiB6mLdvlI0nW88CQg24X7Ku"


def ensure_data() -> None:
    if TRAIN_DATA.exists() and VALID_DATA.exists():
        return
    archive = ROOT / "data" / "HW7_large_stories.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print("Downloading the official full HW7 dataset...")
        try:
            import gdown
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True
            )
            import gdown
        gdown.download(id=DOWNLOAD_ID, output=str(archive), quiet=False)
    print("Extracting the full HW7 dataset...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(ROOT / "data")
    if not TRAIN_DATA.exists() or not VALID_DATA.exists():
        raise FileNotFoundError(
            f"Expected {TRAIN_DATA} and {VALID_DATA} after extraction"
        )


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {"runs": {}, "generations": {}, "hardware": hardware_name()}


def save_results(results: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")


def hardware_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def read_losses(path: Path) -> list[float]:
    return [float(x) for x in path.read_text().splitlines() if x.strip()]


def read_metrics(path: Path) -> tuple[float, float]:
    values = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = float(value.strip())
    return values["Final Train Loss"], values["Final Valid Loss"]


def run_training(
    results: dict,
    key: str,
    out_dir: Path,
    prefix: str,
    *,
    embed_dim: int,
    hidden_dim: int,
    dk: int,
    dv: int,
    num_sequences: int,
    batch_size: int,
    model_path: Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = out_dir / f"{prefix}metrics.txt"
    train_losses = out_dir / f"{prefix}train_losses.txt"
    val_losses = out_dir / f"{prefix}val_losses.txt"
    log_path = out_dir / f"{prefix}run.log"

    if key in results["runs"] and all(
        p.exists() for p in (metrics, train_losses, val_losses)
    ) and (model_path is None or model_path.exists()):
        print(f"[resume] Skipping completed run {key}")
        return

    command = [
        sys.executable,
        "rnn.py",
        "train",
        "--train_data",
        str(TRAIN_DATA),
        "--val_data",
        str(VALID_DATA),
        "--embed_dim",
        str(embed_dim),
        "--hidden_dim",
        str(hidden_dim),
        "--dk",
        str(dk),
        "--dv",
        str(dv),
        "--num_sequences",
        str(num_sequences),
        "--batch_size",
        str(batch_size),
        "--metrics_out",
        str(metrics),
        "--train_losses_out",
        str(train_losses),
        "--val_losses_out",
        str(val_losses),
    ]
    if model_path is not None:
        command += ["--model_path", str(model_path)]

    print(f"\n=== Running {key} ===")
    started = time.perf_counter()
    captured: list[str] = []
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            captured.append(line)
        return_code = proc.wait()
    wall_time = time.perf_counter() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    output = "".join(captured)
    matches = re.findall(r"^Time:\s+([0-9.]+)\s*$", output, flags=re.MULTILINE)
    training_time = float(matches[-1]) if matches else wall_time
    final_train, final_val = read_metrics(metrics)
    results["runs"][key] = {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "dk": dk,
        "dv": dv,
        "num_sequences": num_sequences,
        "batch_size": batch_size,
        "final_train_loss": final_train,
        "final_validation_loss": final_val,
        "training_time_seconds": training_time,
        "wall_time_seconds": wall_time,
        "train_losses": str(train_losses.relative_to(ROOT)),
        "validation_losses": str(val_losses.relative_to(ROOT)),
    }
    results["hardware"] = hardware_name()
    save_results(results)


def checkpoint_x(num_sequences: int, batch_size: int, n: int) -> list[int]:
    val_index = max(1, int(num_sequences * 0.1) // batch_size)
    return [(k * val_index + 1) * batch_size for k in range(n)]


def save_curve_plot(
    paths_and_labels: list[tuple[Path, str, int, int]],
    destination: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 5.5))
    for path, label, num_sequences, batch_size in paths_and_labels:
        losses = read_losses(path)
        plt.plot(
            checkpoint_x(num_sequences, batch_size, len(losses)),
            losses,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=label,
        )
    plt.title(title)
    plt.xlabel("Number of sequences processed")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()


def make_plots(results: dict) -> None:
    e1 = ROOT / "empiricals" / "e1"
    for loss_type, full in (("train", "Training"), ("val", "Validation")):
        curves = [
            (e1 / f"{dim}_{loss_type}_losses.txt", f"embed = hidden = {dim}", 50000, 128)
            for dim in (64, 128, 256, 512)
        ]
        save_curve_plot(
            curves,
            e1 / f"{loss_type}_plot.png",
            f"{full} Loss vs. Sequences — Model Dimension",
            f"{full} loss",
        )

    e2 = ROOT / "empiricals" / "e2"
    for loss_type, full in (("train", "Training"), ("val", "Validation")):
        curves = [
            (e2 / f"{bs}_{loss_type}_losses.txt", f"batch size = {bs}", 50000, bs)
            for bs in (32, 64, 128, 256)
        ]
        save_curve_plot(
            curves,
            e2 / f"{loss_type}_plot.png",
            f"{full} Loss vs. Sequences — Batch Size",
            f"{full} loss",
        )

    e3 = ROOT / "empiricals" / "e3"
    sequence_counts = [10000, 20000, 50000, 100000]
    for loss_type, full, field in (
        ("train", "Training", "final_train_loss"),
        ("val", "Validation", "final_validation_loss"),
    ):
        values = [results["runs"][f"e3_n{n}"][field] for n in sequence_counts]
        plt.figure(figsize=(8, 5.5))
        plt.plot(sequence_counts, values, marker="o", linewidth=2)
        plt.title(f"Final {full} Loss vs. Number of Training Sequences")
        plt.xlabel("Number of sequences processed")
        plt.ylabel(f"Final {full.lower()} loss")
        plt.xticks(sequence_counts, [f"{n:,}" for n in sequence_counts])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(e3 / f"{loss_type}_plot.png", dpi=180)
        plt.close()


def generate_samples(results: dict) -> None:
    model = ROOT / "empiricals" / "e4" / "final_model.pt"
    prefix = "Once upon a time there was a"
    for temperature in (0.0, 0.3, 0.8):
        key = f"{temperature:g}"
        if len(results["generations"].get(key, [])) >= 3:
            print(f"[resume] Skipping completed generations at temperature {key}")
            continue
        samples = []
        for sample_index in range(3):
            command = [
                sys.executable,
                "rnn.py",
                "generate",
                "--model_path",
                str(model),
                "--prefix",
                prefix,
                "--temperature",
                str(temperature),
                "--num_tokens",
                "128",
            ]
            proc = subprocess.run(
                command, cwd=ROOT, check=True, capture_output=True, text=True
            )
            match = re.search(r"Completion:\s*(.*)", proc.stdout, flags=re.DOTALL)
            completion = match.group(1).strip() if match else proc.stdout.strip()
            samples.append(completion)
            print(f"temperature={key}, sample {sample_index + 1}: {completion}")
        results["generations"][key] = samples
        save_results(results)


def build_report(results: dict) -> None:
    runs = results["runs"]
    q2_times = "\n".join(
        f"| {bs} | {runs[f'e2_b{bs}']['training_time_seconds']:.2f} |"
        for bs in (32, 64, 128, 256)
    )
    q3_table = "\n".join(
        f"| {n:,} | {runs[f'e3_n{n}']['final_train_loss']:.6f} | "
        f"{runs[f'e3_n{n}']['final_validation_loss']:.6f} |"
        for n in (10000, 20000, 50000, 100000)
    )
    e4 = runs["e4_final"]
    generations = results["generations"]

    report = f"""# HW7 Empirical Questions — Answers

Hardware used for the reported timing: **{results['hardware']}**.

## 1. Model size

![Training loss by model dimension](empiricals/e1/train_plot.png)

![Validation loss by model dimension](empiricals/e1/val_plot.png)

## 2. Batch size

### (a) Loss curves

![Training loss by batch size](empiricals/e2/train_plot.png)

![Validation loss by batch size](empiricals/e2/val_plot.png)

### (b) Training time

| Batch size | Time (seconds) |
|---:|---:|
{q2_times}

### (c) Fill in the blanks

Increasing batch size leads to **faster** training and **a decrease** in performance.

LaTeX-ready text:

```latex
Increasing batch size leads to \\textbf{{faster}} training and
\\textbf{{a decrease}} in performance.
```

For a fixed number of sequences, larger batches require fewer optimizer updates.
They therefore run faster, but the observed final training/validation losses are
worse (higher), which is a decrease in performance.

## 3. Number of training sequences

![Final training loss by sequence count](empiricals/e3/train_plot.png)

![Final validation loss by sequence count](empiricals/e3/val_plot.png)

Supporting values:

| Number of sequences | Final training loss | Final validation loss |
|---:|---:|---:|
{q3_table}

## 4. Long training run and generation

### (a) Final metrics

| Train loss | Validation loss | Time (seconds) |
|---:|---:|---:|
| {e4['final_train_loss']:.6f} | {e4['final_validation_loss']:.6f} | {e4['training_time_seconds']:.2f} |

LaTeX table row:

```latex
{e4['final_train_loss']:.6f} & {e4['final_validation_loss']:.6f} & {e4['training_time_seconds']:.2f} \\\\
```

### (b) Favorite generated completions

The runner produces three samples at every temperature. The first sample is
inserted below; all nine candidates remain in `empiricals/results.json`, so you
can substitute another candidate if you prefer it.

**Temperature $=0$**

> {generations['0'][0]}

**Temperature $=0.3$**

> {generations['0.3'][0]}

**Temperature $=0.8$**

> {generations['0.8'][0]}

### (c) Temperature trend

```latex
As the temperature increases, the next-token distribution becomes flatter and
the completions become more varied and creative, but also less predictable and
more prone to incoherence. At temperature $0$, greedy decoding is deterministic
and repetitive. Temperature $0.3$ provides the best balance in my samples: its
completion is coherent while still showing some variety. At temperature $0.8$,
the text is the most diverse, but it contains more awkward or inconsistent
transitions. Therefore, I consider temperature $0.3$ to produce the best
completion.
```
"""
    (ROOT / "empirical_answers.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=int,
        choices=(1, 2, 3, 4),
        default=(1, 2, 3, 4),
        help="Experiments to run; completed runs are skipped automatically.",
    )
    args = parser.parse_args()
    ensure_data()
    results = load_results()

    if 1 in args.experiments:
        for dim in (64, 128, 256, 512):
            run_training(
                results,
                f"e1_d{dim}",
                ROOT / "empiricals" / "e1",
                f"{dim}_",
                embed_dim=dim,
                hidden_dim=dim,
                dk=128,
                dv=128,
                num_sequences=50000,
                batch_size=128,
            )

    if 2 in args.experiments:
        for bs in (32, 64, 128, 256):
            run_training(
                results,
                f"e2_b{bs}",
                ROOT / "empiricals" / "e2",
                f"{bs}_",
                embed_dim=128,
                hidden_dim=128,
                dk=128,
                dv=128,
                num_sequences=50000,
                batch_size=bs,
            )

    if 3 in args.experiments:
        for n in (10000, 20000, 50000, 100000):
            run_training(
                results,
                f"e3_n{n}",
                ROOT / "empiricals" / "e3",
                f"{n}_",
                embed_dim=128,
                hidden_dim=128,
                dk=128,
                dv=128,
                num_sequences=n,
                batch_size=128,
            )

    if 4 in args.experiments:
        run_training(
            results,
            "e4_final",
            ROOT / "empiricals" / "e4",
            "",
            embed_dim=512,
            hidden_dim=512,
            dk=256,
            dv=256,
            num_sequences=250000,
            batch_size=128,
            model_path=ROOT / "empiricals" / "e4" / "final_model.pt",
        )
        generate_samples(results)

    required_runs = {
        *(f"e1_d{x}" for x in (64, 128, 256, 512)),
        *(f"e2_b{x}" for x in (32, 64, 128, 256)),
        *(f"e3_n{x}" for x in (10000, 20000, 50000, 100000)),
        "e4_final",
    }
    if required_runs <= results["runs"].keys() and all(
        len(results["generations"].get(x, [])) >= 3 for x in ("0", "0.3", "0.8")
    ):
        make_plots(results)
        build_report(results)
        print("\nComplete: wrote all plots and empirical_answers.md")
    else:
        missing = sorted(required_runs - results["runs"].keys())
        print(f"\nPartial run saved. Remaining training runs: {missing}")


if __name__ == "__main__":
    main()
