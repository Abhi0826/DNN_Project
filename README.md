# APM 527 Final Project — Neural Piano Music Generation from a Procedurally Generated Symbolic Dataset

**Team:** Sai Abhishek Bhattiprolu, Shivam Sharma, Vansh Bharat Gala
**Course:** APM 527: Deep Neural Networks

## Contents

| File | What it is |
|---|---|
| `APM527_Piano_Music_Generation.py` | Complete script: loads the dataset, trains and evaluates all models, exports MIDI and figures |
| `data/dataset_splits.json` | **The dataset**: 4,800 / 600 / 600 train / val / test pieces as token-ID sequences, plus the vocabulary and generator settings |
| `outputs/generated_piano.mid` | Primary generated piece (LSTM, T = 1.0, selected by the objective rule in Section 15) |
| `outputs/generated_piano.wav` | Audio rendering of the MIDI with FluidSynth and the FluidR3_GM General MIDI soundfont |
| `outputs/generated_piano_T{0.7,1.0,1.3}.mid` | First LSTM sample at each temperature, exported without selection |
| `outputs/lstm_best.pt`, `outputs/linear_best.pt` | Checkpoints (best validation loss) |
| `outputs/*.csv`, `outputs/results_summary.json` | Training histories and all reported numbers |
| `outputs/figures/` | Figures produced by the script |
| `report/APM527_Final_Report.tex` | Two-page IEEE conference report (LaTeX source) |
| `report/APM527_Final_Report.pdf` | Compiled report |
| `report/IEEEtran.cls` | IEEE class file (V1.8b), included so the report compiles offline |
| `REQUIREMENTS_AUDIT.md` | Transcription and checklist of the handwritten course instructions |

## Rerunning

1. Python 3.10+ with: `torch numpy pandas matplotlib pretty_midi`
   (`pip install torch numpy pandas matplotlib pretty_midi`). A GPU is optional.
2. Keep the `data/` folder next to the script and run
   `python APM527_Piano_Music_Generation.py`. It can be started from any directory; paths
   resolve relative to the script. On a 2-core CPU the default configuration takes about
   10 minutes. Progress and all metrics are printed to the terminal.
3. Outputs are written to `outputs/` (relative path). Everything is seeded (seed 527); results
   repeat exactly on the same library versions and hardware.
4. WAV rendering is optional and runs only if `fluidsynth` and a General MIDI soundfont are
   installed (set the `SOUNDFONT` environment variable to point at one).
5. Report: `cd report && pdflatex APM527_Final_Report.tex && pdflatex APM527_Final_Report.tex`
   (or upload the `.tex`, `IEEEtran.cls` and `training_curves_report.pdf` from `report/` to Overleaf).

The script reads `data/dataset_splits.json`. If that file is missing, or you pass `--regenerate-data`,
it rebuilds the file with the procedural generator in Section 5 (seed 527 gives an identical file).
`--long-run` enables a larger optional setting; the reported
results all use the default.
