# Run the HW7 empirical questions on Google Colab

The full empirical dataset is not included in the downloaded handout, and the
assignment estimates roughly 60--85 minutes of GPU time for all required runs.
Use a **T4 or A100 GPU**, not a CPU.

1. Upload the entire `handout` folder to your Google Drive.
2. Open `colab_notebook.ipynb` in Colab.
3. Select **Runtime -> Change runtime type -> T4 GPU** (or A100).
4. Run the Drive-mount cell and change-directory cell already in the notebook.
5. In a new code cell, run:

   ```python
   !python run_empiricals.py
   ```

The runner downloads/extracts the official full dataset automatically. It saves
every completed run under `empiricals/`, so if Colab disconnects, rerun the same
command and it will resume. At the end it creates:

- `empirical_answers.md` -- all numeric/text answers in copy-ready form
- `empiricals/e1/train_plot.png` and `val_plot.png`
- `empiricals/e2/train_plot.png` and `val_plot.png`
- `empiricals/e3/train_plot.png` and `val_plot.png`
- `empiricals/results.json` -- all metrics and all nine generated candidates
- `empiricals/e4/final_model.pt` -- the trained model

To split the work across sessions, run selected experiments. Completed runs are
always skipped:

```python
!python run_empiricals.py --experiments 1 2
!python run_empiricals.py --experiments 3 4
```

Keep the Colab tab open while a command is running. Because the folder is on
Drive, outputs survive a runtime restart.
