# HW7: RNN

**Read the programming section of the HW7 writeup before you do anything with
this directory.** It is the only set of instructions for this assignment, and
it covers all of it.

You write one program here, `rnn.py`. The datasets are in `data/`, so the
commands in the writeup take paths like `data/small_train_stories.json`.

## Setup

```
conda env create -f environment.yml
conda activate 601-hw7
```

Keep the earlier assignments' environments; do not delete them to make room.

## Running the graded tests

```
pytest test_rnn_language_model.py
```

(a bare `pytest` runs the same file). The graded tests fall into two groups:
component tests that check individual pieces of `rnn.py` (the RNN cell, the
recurrence, attention, the hybrid language model, ...) against fixed inputs,
and five end-to-end training runs, each on CPU, that together take about 1.5
minutes. Your program runs on CPU during grading even if a GPU is present on
your machine.

## Local unit tests

Before you submit, you can also run the same local unit tests the Gradescope
version of this assignment used:

```
python test_runner.py
```

These are independent, ungraded checks on fixed inputs -- useful for
localizing a bug before running the graded suite.
