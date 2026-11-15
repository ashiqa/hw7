"""PhiGrade test suite for 10-301/601 HW7: RNN language model.

Run this from the directory that contains your rnn.py:

    pytest test_rnn_language_model.py

Two kinds of test. The component tests build each of your modules (RNNCell,
RNN, SelfAttention, HybridLanguageModel), load a fixed set of weights into
it, run it on fixed inputs, and compare what it returns with the reference
solution's output on the same weights and inputs. The metric tests run
`python rnn.py train ...` on five configurations and compare the final losses
it writes. See README.md for setup.
"""

import importlib.util
import math
import os
import re
import subprocess
import sys
import traceback
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

# These tests run on the CPU, as the answer key was recorded: the GPU is
# hidden from torch before it is imported, so DEVICE in your rnn.py is "cpu"
# here even on a machine with a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import phigrade
import torch
from phigrade import weight

SCRIPT = "rnn.py"
TIMEOUT_SECONDS = 300

# Your program runs on the CPU with one thread, as the answer key was
# recorded.
SUBPROCESS_ENV = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                  "MKL_NUM_THREADS": "1"}

TENSOR_RTOL = 1e-5
TENSOR_ATOL = 1e-5

METRIC_RTOL = 1e-6
METRIC_ATOL = 0.0

SUBSAMPLE_ABOVE = 2500
STRIDES = {(5, 20, 100): 7, (5, 20, 32): 3, (8, 50, 50): 11, (8, 50, 16): 5}


class Config(NamedTuple):
    """One `rnn.py train` command line."""

    data: str
    embed_dim: int
    hidden_dim: int
    num_sequences: int
    batch_size: int

    def describe(self) -> str:
        return (f"the {self.data} dataset (embed_dim {self.embed_dim}, "
                f"hidden_dim {self.hidden_dim}, {self.num_sequences} "
                f"sequences, batch size {self.batch_size})")

    def dirname(self) -> str:
        return (f"{self.data}_e{self.embed_dim}_h{self.hidden_dim}"
                f"_n{self.num_sequences}_b{self.batch_size}")


SMOKE = Config("tiny", 32, 32, 16, 1)
TINY_B1 = Config("tiny", 32, 32, 128, 1)
TINY_B16 = Config("tiny", 32, 32, 128, 16)
TINY_H64_B16 = Config("tiny", 32, 64, 128, 16)
SMALL_N400_B10 = Config("small", 128, 128, 400, 10)
SMALL_N512_B32 = Config("small", 128, 128, 512, 32)


class RnnRun(NamedTuple):
    """One run of `rnn.py train`: the files it wrote and what it printed."""

    metrics: str
    train_losses: str
    val_losses: str
    stdout: str
    stderr: str


def _read(path: Path, description: str) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        raise AssertionError(
            f"{SCRIPT} did not create {path} when run on {description}. "
            f"Check that you are writing to the output paths given on the "
            f"command line."
        ) from None


def _run(cmd: list[str], description: str) -> subprocess.CompletedProcess:
    """Run one command; return the completed process."""
    if not Path(SCRIPT).is_file():
        raise AssertionError(
            f"Could not find {SCRIPT} in {Path.cwd()}. "
            f"Run pytest from the directory containing your solution."
        )
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=TIMEOUT_SECONDS,
                              env={**os.environ, **SUBPROCESS_ENV})
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"{SCRIPT} did not finish within {TIMEOUT_SECONDS} seconds on "
            f"{description}. Is there an infinite loop?"
        ) from None

    if proc.returncode != 0:
        raise AssertionError(
            f"{SCRIPT} exited with status {proc.returncode} on {description}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc


@lru_cache(maxsize=None)
def run_rnn(config: Config) -> RnnRun:
    """Run `rnn.py train` on one configuration; return what it wrote.

    Cached, so each configuration costs exactly one subprocess however many
    tests read it. Each writes into its own directory, stale files from an
    earlier pytest run are removed first, and the files are read immediately,
    so no test can observe another run's output.
    """
    description = config.describe()
    outdir = Path("out") / config.dirname()
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_out = outdir / "metrics.txt"
    train_losses_out = outdir / "train_losses.txt"
    val_losses_out = outdir / "val_losses.txt"
    for path in (metrics_out, train_losses_out, val_losses_out):
        path.unlink(missing_ok=True)

    cmd = [
        sys.executable, SCRIPT, "train",
        "--train_data", f"data/{config.data}_train_stories.json",
        "--val_data", f"data/{config.data}_valid_stories.json",
        "--embed_dim", str(config.embed_dim),
        "--hidden_dim", str(config.hidden_dim),
        "--dk", "32", "--dv", "32",
        "--num_sequences", str(config.num_sequences),
        "--metrics_out", str(metrics_out),
        "--train_losses_out", str(train_losses_out),
        "--val_losses_out", str(val_losses_out),
        "--batch_size", str(config.batch_size),
    ]
    proc = _run(cmd, description)

    return RnnRun(
        metrics=_read(metrics_out, description),
        train_losses=_read(train_losses_out, description),
        val_losses=_read(val_losses_out, description),
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


@contextmanager
def _showing_your_output(outputs, description: str):
    """Attach whatever your program printed to a failing comparison.

    A program that crashes already reports its output, because the non-zero
    exit is itself the failure. A program that runs but computes the wrong
    answer is the harder case, and its output would otherwise be thrown away:
    it runs in a subprocess whose output this file captures.
    """
    try:
        yield
    except AssertionError as failure:
        parts = [str(failure)]
        if outputs.stdout.strip():
            parts.append(
                f"--- what your program printed on {description} ---\n"
                f"{outputs.stdout.rstrip()}"
            )
        if outputs.stderr.strip():
            parts.append(f"--- and on stderr ---\n{outputs.stderr.rstrip()}")
        raise AssertionError("\n\n".join(parts)) from None


@lru_cache(maxsize=None)
def student():
    """Your rnn.py, imported once per session from the current directory.

    Called inside each component test rather than at the top of this file, so
    that an rnn.py which fails to import fails those tests with its traceback
    instead of stopping pytest from collecting anything.
    """
    path = Path(SCRIPT)
    if not path.is_file():
        raise AssertionError(
            f"Could not find {SCRIPT} in {Path.cwd()}. "
            f"Run pytest from the directory containing your solution.")
    spec = importlib.util.spec_from_file_location("rnn", path.resolve())
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        raise AssertionError(
            f"Importing your {SCRIPT} raised an exception, so none of its "
            f"classes can be tested:\n\n{traceback.format_exc()}") from None
    return module


# ---------------------------------------------------------------------------
# Deterministic construction.
# ---------------------------------------------------------------------------


def _linear(prefix: str, fan_in: int, fan_out: int) -> list:
    """An nn.Linear(fan_in, fan_out)'s parameters, drawn from the range
    PyTorch itself initialises them in."""
    bound = 1.0 / math.sqrt(fan_in)
    return [(f"{prefix}.weight", (fan_out, fan_in), bound),
            (f"{prefix}.bias", (fan_out,), bound)]


def rnn_cell_layout(input_dim, hidden_dim, prefix=""):
    return (_linear(f"{prefix}i2h", input_dim, hidden_dim)
            + _linear(f"{prefix}h2h", hidden_dim, hidden_dim))


def rnn_layout(input_dim, hidden_dim, prefix=""):
    return (rnn_cell_layout(input_dim, hidden_dim, f"{prefix}cell.")
            + _linear(f"{prefix}out", hidden_dim, hidden_dim))


def attention_layout(hidden_dim, key_dim, value_dim, prefix=""):
    return (_linear(f"{prefix}query_transform", hidden_dim, key_dim)
            + _linear(f"{prefix}key_transform", hidden_dim, key_dim)
            + _linear(f"{prefix}value_transform", hidden_dim, value_dim)
            + _linear(f"{prefix}output_transform", value_dim, hidden_dim))


def language_model_layout(embed_dim, hidden_dim, vocab_size, key_dim, value_dim):
    return ([("embeddings.weight", (vocab_size, embed_dim), 1.0)]
            + rnn_layout(embed_dim, hidden_dim, "rnn.")
            + attention_layout(hidden_dim, key_dim, value_dim, "attention.")
            + _linear("lm_head", hidden_dim, vocab_size))


def uniform(rng, shape, bound) -> torch.Tensor:
    values = rng.uniform(-bound, bound, size=shape).astype(np.float32)
    return torch.from_numpy(values)


def make_parameters(rng, layout) -> dict:
    """Draw every parameter in `layout`, in order. The layout comes from the
    test's dimensions, never from your module, so an extra or misshapen
    parameter of yours cannot change what anyone else's test draws."""
    return {name: uniform(rng, shape, bound) for name, shape, bound in layout}


def make_tokens(rng, vocab_size, shape) -> torch.Tensor:
    return torch.from_numpy(rng.integers(0, vocab_size, size=shape, dtype=np.int64))


def load_parameters(module, params, component) -> None:
    """Load the test's parameters into your module and switch it to eval mode.

    Every parameter the test sets must exist in your module with the shape
    listed. An extra parameter of your own is left untouched, not an error.
    """
    # load_state_dict(strict=False) still raises on a shape mismatch. Its
    # "unexpected" keys are the test's parameters your module lacks; its
    # "missing" keys are your extras, which are allowed.
    try:
        result = module.load_state_dict(params, strict=False)
        problem = None
        if result.unexpected_keys:
            problem = ("Parameters missing from your module: "
                       + ", ".join(result.unexpected_keys))
    except (RuntimeError, AttributeError, TypeError) as error:
        problem = str(error)
    if problem is not None:
        expected = "\n".join(f"    {name}: {tuple(t.shape)}"
                             for name, t in params.items())
        raise AssertionError(
            f"Could not load the test's weights into your {component}:\n"
            f"{problem}\n\n"
            f"The test expects these parameters, with these shapes:\n"
            f"{expected}\n"
            f"Create them with nn.Linear / nn.Embedding under the attribute "
            f"names the starter code gives.") from None
    module.eval()


# ---------------------------------------------------------------------------
# Tensor checks.
# ---------------------------------------------------------------------------


def subsample(values, stride):
    """Every `stride`-th value of a flattened array. One-sided, so the
    reference run and yours sample identically."""
    return np.asarray(values).ravel()[::stride]


def check_tensor(value, name, expected_shape) -> None:
    """Compare one of your tensors with the reference's, as checkpoint `name`.
    """
    expected_shape = tuple(expected_shape)
    stride = 1
    if math.prod(expected_shape) > SUBSAMPLE_ABOVE:
        if expected_shape not in STRIDES:
            raise RuntimeError(
                f"test error: no subsampling stride for shape {expected_shape}")
        stride = STRIDES[expected_shape]
    if not isinstance(value, torch.Tensor):
        raise AssertionError(
            f"{name}: expected a torch.Tensor, got {type(value).__name__}.")
    array = value.detach().cpu().numpy()
    bad = int(array.size - np.count_nonzero(np.isfinite(array)))
    if bad:
        raise AssertionError(
            f"{name}: {bad} of {array.size} values in your output are NaN or "
            f"infinite, so it cannot be compared with the reference.")
    if stride > 1:
        if array.shape != expected_shape:
            raise AssertionError(
                f"{name}: your tensor has shape {array.shape}; shape "
                f"{expected_shape} was expected.")
        array = subsample(array, stride)
    phigrade.all_close(array, rtol=TENSOR_RTOL, atol=TENSOR_ATOL, name=name)


# ---------------------------------------------------------------------------
# One-sided preprocessing: the metrics parser.
# ---------------------------------------------------------------------------

_METRIC_LINE = re.compile(
    r"^\s*Final\s+(?P<split>Train|Valid)\s+Loss\s*:\s*(?P<value>\S+)\s*$")

_METRIC_SHAPES = ("    Final Train Loss: 4.727963\n"
                  "    Final Valid Loss: 4.91959")


def parse_metrics(text: str) -> dict:
    """Parse metrics.txt into {"Final Train Loss": ..., "Final Valid Loss": ...}.
    """
    values = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            raise AssertionError(
                f"Line {number} of your metrics file is not in a recognised "
                f"format:\n    {line!r}\n"
                f"Every line should look like one of these:\n{_METRIC_SHAPES}")
        key = f"Final {match['split']} Loss"
        try:
            value = float(match["value"])
        except ValueError:
            raise AssertionError(
                f"Line {number} of your metrics file does not end in a "
                f"number:\n    {line!r}\n"
                f"Could not read {match['value']!r} as a number.") from None
        if not math.isfinite(value):
            raise AssertionError(
                f"Line {number} of your metrics file: {match['value']!r} is "
                f"not a finite number:\n    {line!r}\n"
                f"Your training diverged or produced NaN.")
        if key in values:
            raise AssertionError(
                f"Line {number} of your metrics file repeats {key!r}, which "
                f"an earlier line already gave. Write each loss exactly once.")
        values[key] = value
    return values


# ---------------------------------------------------------------------------
# Component checks. Each builds your module, loads the test's weights, runs it
# on the test's inputs, and compares each output as a named checkpoint.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _check_rnn_cell(seed, input_dim, hidden_dim, batch_size=4):
    rng = np.random.default_rng(seed)
    cell = student().RNNCell(input_dim, hidden_dim)
    load_parameters(cell, make_parameters(rng, rnn_cell_layout(input_dim, hidden_dim)), "RNNCell")
    x = uniform(rng, (batch_size, input_dim), 1.0)
    h = uniform(rng, (batch_size, hidden_dim), 1.0)
    check_tensor(cell(x, h), "hidden", (batch_size, hidden_dim))


@torch.no_grad()
def _check_rnn_step(seed, input_dim, hidden_dim, batch_size):
    rng = np.random.default_rng(seed)
    rnn = student().RNN(input_dim, hidden_dim)
    load_parameters(rnn, make_parameters(rng, rnn_layout(input_dim, hidden_dim)), "RNN")
    x = uniform(rng, (batch_size, input_dim), 1.0)
    hidden_prev = uniform(rng, (batch_size, 1, hidden_dim), 1.0)
    hidden, output = rnn.step(x, hidden_prev)
    check_tensor(hidden, "hidden", (batch_size, hidden_dim))
    check_tensor(output, "output", (batch_size, hidden_dim))


@torch.no_grad()
def _check_rnn_forward(seed, input_dim, hidden_dim, batch_size, seq_len):
    rng = np.random.default_rng(seed)
    rnn = student().RNN(input_dim, hidden_dim)
    load_parameters(rnn, make_parameters(rng, rnn_layout(input_dim, hidden_dim)), "RNN")
    sequence = uniform(rng, (batch_size, seq_len, input_dim), 1.0)
    hidden_states, output_states = rnn(sequence)
    check_tensor(hidden_states, "hidden_states", (batch_size, seq_len, hidden_dim))
    check_tensor(output_states, "output_states", (batch_size, seq_len, hidden_dim))


def _attention(rng, embed_dim, key_dim, value_dim):
    attention = student().SelfAttention(embed_dim, key_dim, value_dim)
    load_parameters(attention, make_parameters(
        rng, attention_layout(embed_dim, key_dim, value_dim)), "SelfAttention")
    return attention


@torch.no_grad()
def _check_attention_step(seed, embed_dim, key_dim, value_dim, batch_size, seq_len):
    """One checkpoint per prefix length t, named `t=<t>`.
    """
    rng = np.random.default_rng(seed)
    attention = _attention(rng, embed_dim, key_dim, value_dim)
    y_all = uniform(rng, (batch_size, seq_len, embed_dim), 1.0)
    for t in range(1, seq_len + 1):
        check_tensor(attention.step(y_all[:, :t]), f"t={t}", (batch_size, embed_dim))


@torch.no_grad()
def _check_attention_forward(seed, embed_dim, key_dim, value_dim, batch_size, seq_len):
    rng = np.random.default_rng(seed)
    attention = _attention(rng, embed_dim, key_dim, value_dim)
    y_all = uniform(rng, (batch_size, seq_len, embed_dim), 1.0)
    check_tensor(attention(y_all), "output", (batch_size, seq_len, embed_dim))


def _language_model(rng, embed_dim, hidden_dim, vocab_size, key_dim, value_dim):
    lm = student().HybridLanguageModel(embed_dim, hidden_dim, vocab_size, key_dim, value_dim)
    load_parameters(lm, make_parameters(rng, language_model_layout(
        embed_dim, hidden_dim, vocab_size, key_dim, value_dim)), "HybridLanguageModel")
    return lm


@torch.no_grad()
def _check_language_model_step(seed, embed_dim, hidden_dim, vocab_size, key_dim, value_dim):
    """One generation step from a single token, checked stage by stage.
    """
    rng = np.random.default_rng(seed)
    lm = _language_model(rng, embed_dim, hidden_dim, vocab_size, key_dim, value_dim)
    token = make_tokens(rng, vocab_size, (1,))
    hidden, attention_input = lm.rnn.step(lm.embeddings(token), None)
    check_tensor(hidden, "hidden", (1, hidden_dim))
    output = lm.attention.step(attention_input.unsqueeze(1))
    check_tensor(output, "attention_output", (1, hidden_dim))
    check_tensor(lm.lm_head(output), "logits", (1, vocab_size))


@torch.no_grad()
def _check_language_model_forward(seed, embed_dim, hidden_dim, vocab_size, key_dim,
                                  value_dim, batch_size, seq_len):
    rng = np.random.default_rng(seed)
    lm = _language_model(rng, embed_dim, hidden_dim, vocab_size, key_dim, value_dim)
    tokens = make_tokens(rng, vocab_size, (batch_size, seq_len))
    token_logits, hidden_states, attention_inputs = lm(tokens)
    check_tensor(token_logits, "token_logits", (batch_size, seq_len, vocab_size))
    check_tensor(hidden_states, "hidden_states", (batch_size, seq_len, hidden_dim))
    check_tensor(attention_inputs, "attention_inputs", (batch_size, seq_len, hidden_dim))


def _check_metrics(config):
    run = run_rnn(config)
    with _showing_your_output(run, config.describe()):
        phigrade.mapping_match(parse_metrics(run.metrics),
                               rtol=METRIC_RTOL, atol=METRIC_ATOL)


# ---------------------------------------------------------------------------
# The graded tests.
# ---------------------------------------------------------------------------


@weight(2.0)
def test_program_runs():
    """rnn.py trains to completion and writes all three output files."""
    run_rnn(SMOKE)
    phigrade.always_pass()


@weight(2.0)
def test_rnn_cell_1():
    _check_rnn_cell(7001, input_dim=32, hidden_dim=16)


@weight(2.0)
def test_rnn_cell_2():
    _check_rnn_cell(7002, input_dim=64, hidden_dim=32)


@weight(2.0, aggregation="even_weight")
def test_rnn_step_1():
    _check_rnn_step(7003, input_dim=64, hidden_dim=32, batch_size=4)


@weight(2.0, aggregation="even_weight")
def test_rnn_step_2():
    _check_rnn_step(7004, input_dim=32, hidden_dim=16, batch_size=8)


@weight(2.0, aggregation="even_weight")
def test_rnn_forward_1():
    _check_rnn_forward(7005, input_dim=64, hidden_dim=32, batch_size=5, seq_len=20)


@weight(2.0, aggregation="even_weight")
def test_rnn_forward_2():
    _check_rnn_forward(7006, input_dim=32, hidden_dim=16, batch_size=8, seq_len=50)


@weight(2.0, aggregation="even_weight")
def test_attention_step_1():
    _check_attention_step(7007, 64, 64, 64, batch_size=3, seq_len=12)


@weight(2.0, aggregation="even_weight")
def test_attention_step_2():
    _check_attention_step(7008, 32, 32, 32, batch_size=4, seq_len=8)


@weight(2.0)
def test_attention_forward_1():
    _check_attention_forward(7009, 64, 64, 64, batch_size=3, seq_len=12)


@weight(2.0)
def test_attention_forward_2():
    _check_attention_forward(7010, 32, 32, 32, batch_size=4, seq_len=8)


@weight(2.0, aggregation="even_weight")
def test_language_model_step_1():
    _check_language_model_step(7011, 64, 32, 100, 32, 32)


@weight(2.0, aggregation="even_weight")
def test_language_model_step_2():
    _check_language_model_step(7012, 32, 16, 50, 16, 16)


@weight(2.0, aggregation="even_weight")
def test_language_model_forward_1():
    _check_language_model_forward(7013, 64, 32, 100, 32, 32, batch_size=5, seq_len=20)


@weight(2.0, aggregation="even_weight")
def test_language_model_forward_2():
    _check_language_model_forward(7014, 32, 16, 50, 16, 16, batch_size=8, seq_len=50)


@weight(3.2)
def test_metrics_tiny_e32_h32_n128_b1():
    _check_metrics(TINY_B1)


@weight(3.2)
def test_metrics_tiny_e32_h32_n128_b16():
    _check_metrics(TINY_B16)


@weight(3.2)
def test_metrics_tiny_e32_h64_n128_b16():
    _check_metrics(TINY_H64_B16)


@weight(3.2)
def test_metrics_small_e128_h128_n400_b10():
    _check_metrics(SMALL_N400_B10)


@weight(3.2)
def test_metrics_small_e128_h128_n512_b32():
    _check_metrics(SMALL_N512_B32)
