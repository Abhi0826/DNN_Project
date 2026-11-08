#!/usr/bin/env python3
"""
Neural Piano Music Generation from a Procedurally Generated Symbolic Dataset
APM 527: Deep Neural Networks - Final Project
Team: Sai Abhishek Bhattiprolu, Shivam Sharma, Vansh Bharat Gala

This script loads the project dataset (data/dataset_splits.json, produced by the generator
in Section 5), trains every model, evaluates them, and writes the generated piano MIDI file.
All outputs, including figures, are written under outputs/ next to this script.

Usage:
    pip install torch numpy pandas matplotlib pretty_midi
    python APM527_Piano_Music_Generation.py                    # reported configuration
    python APM527_Piano_Music_Generation.py --regenerate-data  # rebuild data/ from the generator
    python APM527_Piano_Music_Generation.py --long-run         # optional larger setting

No GPU is required; the default configuration takes about 10 minutes on a 2-core CPU.
"""

# ============================================================================================
# 1. Project objective
# ============================================================================================
#
# Research question. Can a compact recurrent network, trained only on procedurally generated
# monophonic piano melodies, learn the local melodic and rhythmic regularities of that data well
# enough to (a) predict held-out sequences better than linear and Markov baselines and
# (b) sample new, syntactically valid, bar-aligned melodies that can be written to MIDI?
#
# What the experiment can and cannot show. All training data comes from a rule-based
# generator, so success means the model reproduces the statistics of *those rules*. It says
# nothing direct about human-composed music, and objective metrics below are not a measure of
# musical quality. The generated MIDI is a qualitative artifact for listening, not evidence.

# ============================================================================================
# 2. Interpretation of the course requirements (handwritten Lecture 9 notes)
# ============================================================================================
#
# | PDF requirement | Where it is satisfied |
# |---|---|
# | Use a dataset, *downloaded or generated* | Section 5: procedurally generated dataset, stored in `data/dataset_splits.json` and loaded by this script |
# | Show training for both train and test sets (loss vs. epochs) | Section 11: per-epoch train / validation / test loss for the LSTM and the linear model |
# | At least one baseline, e.g. compare with a linear model; do not just train one deep network | Sections 9–10: linear softmax model on a one-hot context window, bigram Markov chain, unigram reference |
# | Example project 5: "Generate music (piano) → RNN" | Main model is an LSTM language model over note events |
#
# Notes: the PDF crosses out "report" and specifies a 25 + 5 minute presentation; the team's
# two-page IEEE report is produced separately. The test-set curve is monitored only;
# early stopping and checkpoint selection use the validation set, so the test set never
# influences training. A full requirements audit is in `REQUIREMENTS_AUDIT.md`.

# ============================================================================================
# 3. Environment setup and imports
# ============================================================================================
#
# Install missing packages with: pip install torch numpy pandas matplotlib pretty_midi
# No GPU is required; the default configuration was run on a 2-core CPU.

import os, sys, json, math, time, random, shutil, subprocess, platform, argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed; figures are saved to outputs/figures
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretty_midi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Python      {platform.python_version()}")
for mod in (np, pd, matplotlib, torch):
    print(f"{mod.__name__:<11} {mod.__version__}")
print(f"pretty_midi {getattr(pretty_midi, '__version__', 'installed')}")
print(f"Device      {DEVICE}")

parser = argparse.ArgumentParser(description="APM 527 piano music generation")
parser.add_argument("--regenerate-data", action="store_true",
                    help="rebuild data/dataset_splits.json with the procedural generator")
parser.add_argument("--long-run", action="store_true",
                    help="optional larger model and more epochs (not used for reported results)")
ARGS = parser.parse_args()

# Resolve data/ and outputs/ next to this file, wherever it is run from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================================
# 4. Reproducibility configuration
# ============================================================================================
#
# Every hyperparameter lives in `CONFIG`. `--long-run` enables an optional longer
# configuration (larger model, more epochs) for a GPU runtime; all reported results use the
# default configuration.

LONG_RUN = ARGS.long_run

CONFIG = dict(
    seed=527,
    # --- procedural dataset ---
    n_pieces=6000,             # pieces generated before de-duplication
    phrases_per_piece=(2, 3, 4),  # 4-bar phrases -> 8, 12 or 16 bars
    pitch_min=48, pitch_max=84,   # C3..C6
    split=(0.8, 0.1, 0.1),
    # --- training (shared) ---
    batch_size=64,
    grad_clip=1.0,
    patience=3,                # early stopping on validation loss
    # --- LSTM ---
    lstm_embed=64, lstm_hidden=192, lstm_layers=2, lstm_dropout=0.2,
    lstm_lr=2e-3, lstm_max_epochs=40,
    # --- linear baseline ---
    linear_context=8, linear_lr=5e-3, linear_max_epochs=40,
    # --- generation ---
    gen_n_samples=300, gen_max_len=400,
    temperatures=(0.7, 1.0, 1.3),
    primary_temperature=1.0,
    tempo_bpm=90, velocity=80,
)
if LONG_RUN:
    CONFIG.update(lstm_hidden=384, lstm_max_epochs=80, patience=5, n_pieces=12000)

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
torch.use_deterministic_algorithms(True, warn_only=True)
if torch.cuda.is_available():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
print(json.dumps(CONFIG, indent=1))

# ============================================================================================
# 5. Dataset: procedural generator and data file
# ============================================================================================
#
# The generator writes monophonic melodies in 4/4, using 16th-note time units (16 per bar).
# Nothing is downloaded and no existing melody is reproduced. Rules:
#
# - Key and mode: tonic drawn uniformly from the 12 pitch classes; major or natural minor
#   (50/50). In minor, the leading tone is raised at the final cadence.
# - Register: tonic placed between MIDI 60 and 71; every note clipped to MIDI 48–84 by
#   reflecting the melody.
# - Rhythm: each bar uses one of 13 rhythm templates (eighths, quarters, dotted values,
#   halves, and quarter/half rests); cadence bars use 5 templates ending on a long note.
#   Bars are always filled exactly, so no note crosses a bar line.
# - Melodic motion: scale-degree random walk with step probabilities favouring ±1
#   (27% each), ±2 (12%), repeated notes (6%), and bounded leaps of ±3 or ±4 degrees
#   (8% and 6% in total). After a leap, the next step reverses direction with probability 0.7.
# - Motifs and arpeggios: each piece has a one-bar motif (25% of motifs are arpeggiated);
#   the motif repeats as a sequence 1–2 degrees higher or lower. Contrasting phrases open with
#   arpeggios on I, IV, V or vi.
# - Form: 2, 3 or 4 four-bar phrases (8, 12 or 16 bars). Inner phrases end on a half
#   cadence (degree 5), the last phrase restates the opening motif and ends on the tonic.
#
# Representation. Each piece becomes event tokens:
# `START`, then for each note `NOTE_<midi pitch>` followed by `DUR_<16ths>`, or `REST_<16ths>`
# for a rest, then `END`. Velocity is not modelled (fixed at 80 in the MIDI file).

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]       # natural minor
BAR = 16                              # 16th notes per 4/4 bar

# Rhythm templates: positive = note length, negative = rest length (16th-note units)
RHYTHMS = [
    [4, 4, 4, 4], [2, 2, 4, 4, 4], [4, 4, 2, 2, 4], [6, 2, 4, 4], [4, 2, 2, 8],
    [8, 4, 4], [4, 4, 8], [2, 2, 2, 2, 4, 4], [6, 2, 6, 2], [4, -4, 4, 4],
    [8, -4, 4], [4, 4, 4, -4], [2, 2, 2, 2, 2, 2, 4],
]
CADENCE_RHYTHMS = [[16], [8, 8], [4, 4, 8], [12, 4], [4, 12]]   # last value = cadence note
for r in RHYTHMS + CADENCE_RHYTHMS:
    assert sum(abs(v) for v in r) == BAR, r

STEPS = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
STEP_P = np.array([.03, .04, .12, .27, .06, .27, .12, .04, .05])
STEP_P = STEP_P / STEP_P.sum()


class Key:
    """Maps scale degrees (integers, may span octaves) to MIDI pitches."""
    def __init__(self, tonic_midi, mode, pmin, pmax):
        self.tonic, self.mode = tonic_midi, mode
        self.scale = MAJOR if mode == "major" else MINOR
        degs = range(-21, 22)
        ok = [d for d in degs if pmin <= self.pitch(d) <= pmax]
        self.lo, self.hi = min(ok), max(ok)

    def pitch(self, deg, leading_tone=False):
        p = self.tonic + self.scale[deg % 7] + 12 * (deg // 7)
        if leading_tone and self.mode == "minor" and deg % 7 == 6:
            p += 1                              # raised 7th in minor cadences
        return p

    def clip(self, deg):
        while deg < self.lo: deg += 7
        while deg > self.hi: deg -= 7
        return deg


def walk(rng, key, cur, n, prev_step=0):
    """Scale-degree random walk with bounded leaps and leap recovery."""
    out = []
    for _ in range(n):
        if abs(prev_step) >= 3 and rng.random() < 0.7:
            step = -int(np.sign(prev_step))
        else:
            step = int(rng.choice(STEPS, p=STEP_P))
        nxt = cur + step
        if nxt < key.lo or nxt > key.hi:        # reflect at the register boundary
            nxt = cur - step
        nxt = min(max(nxt, key.lo), key.hi)
        prev_step, cur = nxt - cur, nxt
        out.append(cur)
    return out


def make_motif(rng):
    rhythm = RHYTHMS[rng.integers(len(RHYTHMS))]
    n = sum(1 for v in rhythm if v > 0)
    if rng.random() < 0.25:                     # arpeggiated motif on the tonic triad
        shape = [0, 2, 4, 7, 4, 2, 0, 2]
        if rng.random() < 0.5: shape = [-s for s in shape]
        rel = shape[:n]
    else:
        tmp = Key(60, "major", 0, 127)
        rel = [0] + [d for d in walk(rng, tmp, 0, n - 1)]
    return rhythm, rel


def realize(rhythm, degrees, key, leading_last=False):
    """Pair a rhythm with scale degrees -> list of (pitch or None, duration)."""
    ev, i = [], 0
    notes_total = sum(1 for v in rhythm if v > 0)
    for v in rhythm:
        if v < 0:
            ev.append((None, -v))
        else:
            lt = leading_last and i == notes_total - 2
            ev.append((key.pitch(degrees[i], leading_tone=lt), v))
            i += 1
    return ev


def place_motif(rel, start, key):
    degs = [start + r for r in rel]
    while min(degs) < key.lo: degs = [d + 7 for d in degs]
    while max(degs) > key.hi: degs = [d - 7 for d in degs]
    return degs


def nearest_with_class(cur, target_class):
    cands = [d for d in range(cur - 7, cur + 8) if d % 7 == target_class]
    return min(cands, key=lambda d: (abs(d - cur), d))


def cadence_bar(rng, key, cur, final):
    rhythm = CADENCE_RHYTHMS[rng.integers(len(CADENCE_RHYTHMS))]
    n = sum(1 for v in rhythm if v > 0)
    degs = walk(rng, key, cur, n)
    target = key.clip(nearest_with_class(degs[-2] if n > 1 else cur, 0 if final else 4))
    degs[-1] = target
    if n > 1:                                   # approach by step (2nd or leading tone)
        degs[-2] = key.clip(target + (1 if rng.random() < 0.5 else -1))
    return realize(rhythm, degs, key, leading_last=final), degs[-1]


def free_bar(rng, key, cur):
    rhythm = RHYTHMS[rng.integers(len(RHYTHMS))]
    n = sum(1 for v in rhythm if v > 0)
    degs = walk(rng, key, cur, n)
    return realize(rhythm, degs, key), degs[-1]


def arpeggio_bar(rng, key, cur):
    rhythm = RHYTHMS[rng.integers(len(RHYTHMS))]
    n = sum(1 for v in rhythm if v > 0)
    root = int(rng.choice([0, 3, 4, 5]))        # I, IV, V, vi
    tones = [root, root + 2, root + 4, root + 7]
    base = nearest_with_class(cur, root % 7) - root
    seq = (tones + tones[-2:0:-1]) * 3
    if rng.random() < 0.5: seq = seq[::-1]
    degs = [key.clip(base + t) for t in seq[:n]]
    return realize(rhythm, degs, key), degs[-1]


def make_piece(rng, cfg):
    mode = "major" if rng.random() < 0.5 else "minor"
    key = Key(60 + int(rng.integers(12)), mode, cfg["pitch_min"], cfg["pitch_max"])
    n_phr = int(rng.choice(cfg["phrases_per_piece"]))
    motif_r, motif_rel = make_motif(rng)
    start = key.clip(int(rng.choice([0, 2, 4])))
    events, cur = [], start

    def phrase_A(cur, final, variant=False):
        rel = list(motif_rel)
        if variant and len(rel) > 1:            # vary one motif note
            j = int(rng.integers(1, len(rel)))
            rel[j] += int(rng.choice([-1, 1]))
        ev = []
        d1 = place_motif(rel, start, key)
        ev += realize(motif_r, d1, key)
        shift = int(rng.choice([-2, -1, 1, 2]))
        d2 = place_motif(rel, d1[0] + shift, key)
        ev += realize(motif_r, d2, key)
        e3, c = free_bar(rng, key, d2[-1]); ev += e3
        e4, c = cadence_bar(rng, key, c, final); ev += e4
        return ev, c

    def phrase_B(cur):
        ev = []
        e, c = arpeggio_bar(rng, key, cur); ev += e
        e, c = arpeggio_bar(rng, key, c);   ev += e
        e, c = free_bar(rng, key, c);       ev += e
        e, c = cadence_bar(rng, key, c, final=False); ev += e
        return ev, c

    plan = {2: ["A", "Afinal"], 3: ["A", "B", "Afinal"],
            4: ["A", rng.choice(["A'", "B"]), "B", "Afinal"]}[n_phr]
    for p in plan:
        if p == "A":       ev, cur = phrase_A(cur, final=False)
        elif p == "A'":    ev, cur = phrase_A(cur, final=False, variant=True)
        elif p == "B":     ev, cur = phrase_B(cur)
        else:              ev, cur = phrase_A(cur, final=True)
        events += ev
    meta = dict(tonic_pc=key.tonic % 12, mode=mode, bars=4 * n_phr)
    return events, meta


# ---- vocabulary -------------------------------------------------------------
NOTE_DURS = [2, 4, 6, 8, 12, 16]
REST_DURS = [4, 8]
ITOS = (["PAD", "START", "END"]
        + [f"NOTE_{p}" for p in range(CONFIG["pitch_min"], CONFIG["pitch_max"] + 1)]
        + [f"DUR_{d}" for d in NOTE_DURS] + [f"REST_{d}" for d in REST_DURS])
STOI = {s: i for i, s in enumerate(ITOS)}
PAD, START, END = STOI["PAD"], STOI["START"], STOI["END"]
V = len(ITOS)


def events_to_ids(events):
    ids = [START]
    for p, d in events:
        if p is None:
            ids.append(STOI[f"REST_{d}"])
        else:
            ids += [STOI[f"NOTE_{p}"], STOI[f"DUR_{d}"]]
    return ids + [END]


def parse_ids(ids):
    """Lenient parser: returns (events, n_invalid_tokens, n_tokens, ended).
    Grammar: START (NOTE DUR | REST)* END. Tokens violating it are counted as invalid
    and skipped; a NOTE not followed by DUR is dropped."""
    events, invalid, pending, ended, n = [], 0, None, False, 0
    body = ids[1:] if ids and ids[0] == START else ids
    for t in body:
        n += 1
        s = ITOS[t]
        if s == "END":
            if pending is not None: invalid += 1
            ended = True
            break
        if s.startswith("NOTE_"):
            if pending is not None: invalid += 1
            pending = int(s[5:])
        elif s.startswith("DUR_"):
            if pending is None: invalid += 1
            else: events.append((pending, int(s[4:]))); pending = None
        elif s.startswith("REST_"):
            if pending is not None: invalid += 1; pending = None
            events.append((None, int(s[5:])))
        else:                                   # PAD / START mid-sequence
            invalid += 1
    return events, invalid, n, ended


def gen_signature(events, meta):
    """Transposition-invariant identity of a generated piece (used for de-duplication)."""
    notes = [p for p, _ in events if p is not None]
    ivs = tuple(np.diff(notes).tolist())
    rhythm = tuple(d if p is not None else -d for p, d in events)
    return (meta["mode"], (notes[0] - meta["tonic_pc"]) % 12, ivs, rhythm)


def build_dataset(cfg):
    """Generate, validate, de-duplicate and split the dataset. This is the code that
    produced data/dataset_splits.json; it only runs if the file is missing or
    --regenerate-data is given."""
    gen_rng = np.random.default_rng(cfg["seed"])
    raw = [make_piece(gen_rng, cfg) for _ in range(cfg["n_pieces"])]
    for ev, meta in raw:
        assert sum(d for _, d in ev) == BAR * meta["bars"], "bars not filled exactly"
    seen, uniq = set(), []
    for ev, meta in raw:                        # drop transposed duplicates before splitting
        sig = gen_signature(ev, meta)
        if sig not in seen:
            seen.add(sig); uniq.append(ev)
    seqs = [events_to_ids(ev) for ev in uniq]
    perm = np.random.default_rng(cfg["seed"] + 1).permutation(len(seqs))
    n_tr, n_va = int(cfg["split"][0] * len(seqs)), int(cfg["split"][1] * len(seqs))
    idx = dict(train=perm[:n_tr], val=perm[n_tr:n_tr + n_va], test=perm[n_tr + n_va:])
    print(f"Generated {len(raw)} pieces, {len(uniq)} unique after de-duplication")
    return dict(vocab=ITOS, config=cfg, **{k: [seqs[i] for i in v] for k, v in idx.items()})

# --- Load the dataset ---
#
# The dataset used for every result is the file `data/dataset_splits.json`. It holds the
# vocabulary, the generator settings, and the train / validation / test pieces as token-ID
# sequences (4,800 / 600 / 600 pieces).
#
# The script loads this file. If the file is missing, or `--regenerate-data` is given, it is
# rebuilt with the generator above; with seed 527 the rebuilt file is identical.

DATA_FILE = os.path.join("data", "dataset_splits.json")
REGENERATE_DATA = ARGS.regenerate_data

if REGENERATE_DATA or not os.path.exists(DATA_FILE):
    data = build_dataset(CONFIG)
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)
    print(f"Saved dataset to {DATA_FILE}")
else:
    with open(DATA_FILE) as f:
        data = json.load(f)
    print(f"Loaded dataset from {DATA_FILE}")

assert data["vocab"] == ITOS, "dataset vocabulary does not match the script vocabulary"
for k in ("seed", "n_pieces", "phrases_per_piece", "pitch_min", "pitch_max", "split"):
    have, want = json.loads(json.dumps(data["config"][k])), json.loads(json.dumps(CONFIG[k]))
    if have != want:
        print(f"Note: dataset was built with {k}={have} (CONFIG has {want})")
split_seqs = {k: data[k] for k in ("train", "val", "test")}

# Validate every sequence: grammar, pitch range, positive durations, whole bars, round trip
split_events = {}
for k, seq_list in split_seqs.items():
    evs = []
    for s in seq_list:
        ev, bad, _, ended = parse_ids(s)
        notes = [p for p, _ in ev if p is not None]
        assert s[0] == START and bad == 0 and ended, "malformed sequence"
        assert all(CONFIG["pitch_min"] <= p <= CONFIG["pitch_max"] for p in notes), "illegal pitch"
        assert all(d > 0 for _, d in ev), "non-positive duration"
        assert sum(d for _, d in ev) % BAR == 0, "bars not filled exactly"
        assert events_to_ids(ev) == s, "round-trip failed"
        evs.append(ev)
    split_events[k] = evs
print(f"All {sum(len(v) for v in split_seqs.values())} sequences valid; vocabulary size V = {V}")

# ============================================================================================
# 6. Exploratory dataset analysis
# ============================================================================================

seqs = split_seqs["train"] + split_seqs["val"] + split_seqs["test"]
all_events = split_events["train"] + split_events["val"] + split_events["test"]
df = pd.DataFrame([dict(tokens=len(s), notes=sum(p is not None for p, _ in ev),
                        rests=sum(p is None for p, _ in ev), bars=sum(d for _, d in ev) // BAR)
                   for s, ev in zip(seqs, all_events)])
all_notes = [p for ev in all_events for p, _ in ev if p is not None]
all_durs = [d for ev in all_events for p, d in ev if p is not None]

print(f"Pieces: {len(seqs)}   total tokens: {df.tokens.sum():,}   total notes: {len(all_notes):,}")
print(df[["tokens", "notes", "rests", "bars"]].describe().round(1).to_string())
print("\nBars per piece:", df.bars.value_counts().sort_index().to_dict())

fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
ax[0].hist(all_notes, bins=np.arange(CONFIG["pitch_min"] - 0.5, CONFIG["pitch_max"] + 1.5), color="#4C72B0")
ax[0].set(title="Pitch distribution", xlabel="MIDI pitch", ylabel="count")
dc = Counter(all_durs)
ax[1].bar([str(d) for d in NOTE_DURS], [dc[d] for d in NOTE_DURS], color="#4C72B0")
ax[1].set(title="Note durations", xlabel="length (16th notes)")
ax[2].hist(df.tokens, bins=40, color="#4C72B0")
ax[2].set(title="Sequence length", xlabel="tokens per piece")
plt.tight_layout(); plt.savefig(f"{FIG}/dataset_distributions.png", dpi=150); plt.close()

def pretty(events, limit=24):
    names = {2: "e", 4: "q", 6: "q.", 8: "h", 12: "h.", 16: "w"}
    out = []
    for p, d in events[:limit]:
        out.append(f"rest/{names.get(d, d)}" if p is None
                   else f"{pretty_midi.note_number_to_name(p)}/{names.get(d, d)}")
    return " ".join(out) + (" ..." if len(events) > limit else "")

print("Durations: e=eighth q=quarter q.=dotted quarter h=half h.=dotted half w=whole\n")
for i in range(4):
    ev = split_events["train"][i]
    print(f"[train {i}] {sum(d for _, d in ev) // BAR} bars: {pretty(ev)}")
print("\nFirst 20 tokens of train piece 0:", [ITOS[t] for t in split_seqs["train"][0][:20]])

# Is the dataset large enough? There are 6,000 unique pieces (about 0.53 M tokens) over a
# 48-token vocabulary, of which about 0.43 M tokens are used for training. The LSTM has about
# 0.5 M parameters, i.e. slightly more parameters than training tokens, so over-fitting is a
# real risk. Dropout, early stopping on validation loss, and the train/validation gap shown in
# Section 11 are how it is controlled and checked. Because every piece comes from a bounded rule
# set, the data is far more regular than a natural corpus of the same size.
#
# Bias introduced by the rules. Every piece is diatonic, monophonic, in 4/4, bar-aligned,
# uses a fixed set of 13 bar rhythms, and has a phrase form decided in advance. A model trained
# on this data can only learn those conventions; it will not learn harmony, dynamics,
# expressive timing, or anything outside the rule set. This is synthetic data and should not be
# read as representative of human-composed piano music.

# ============================================================================================
# 7. Train / validation / test split (80 / 10 / 10)
# ============================================================================================
# The split is stored in the data file. It was made at the piece level after removing
# transposed duplicates. The check below confirms that no melody appears in two splits in any
# transposition, using a transposition-invariant signature (interval sequence + rhythm).

def signature(events):
    notes = [p for p, _ in events if p is not None]
    return (tuple(np.diff(notes).tolist()), tuple(d if p is not None else -d for p, d in events))

for k in split_seqs:
    assert len(split_seqs[k]) > 0, f"empty {k} split"
sigs = {k: {signature(ev) for ev in v} for k, v in split_events.items()}
assert not (sigs["train"] & sigs["test"]) and not (sigs["train"] & sigs["val"]) \
    and not (sigs["val"] & sigs["test"]), "leakage between splits"
print("No melody (in any transposition) is shared between splits.")
for k, s in split_seqs.items():
    print(f"{k:<5} pieces={len(s):5d}  tokens={sum(len(x) for x in s):8,d}  "
          f"predicted tokens={sum(len(x) - 1 for x in s):8,d}")

# ============================================================================================
# 8. Tokenization and batching
# ============================================================================================
# Each piece is a training example: input = tokens `0..n-2`, target = tokens `1..n-1`.
# Batches group pieces of similar length to reduce padding; `PAD` (id 0) is ignored in the loss.

def make_batches(seq_list, batch_size, shuffle, rng=None):
    order = np.argsort([len(s) for s in seq_list], kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    for b in batches:
        L = max(len(seq_list[i]) for i in b)
        x = torch.full((len(b), L - 1), PAD, dtype=torch.long)
        y = torch.full((len(b), L - 1), PAD, dtype=torch.long)
        for r, i in enumerate(b):
            s = torch.tensor(seq_list[i])
            x[r, :len(s) - 1], y[r, :len(s) - 1] = s[:-1], s[1:]
        yield x.to(DEVICE), y.to(DEVICE)

xb, yb = next(make_batches(split_seqs["train"], 4, False))
assert xb.shape == yb.shape and xb.dim() == 2
m = xb[0, 1:] != PAD                          # inputs are targets shifted by one step
assert torch.equal(xb[0, 1:][m], yb[0, :-1][m])
print("example batch x/y shape:", tuple(xb.shape))

# ============================================================================================
# 9. Models
# ============================================================================================
#
# 1. Unigram (reference). Token frequencies only; ignores context.
# 2. Bigram Markov chain (baseline). $p(x_t \mid x_{t-1})$ from add-α counts; α chosen on validation.
# 3. Linear model (baseline, as suggested in the course notes). A softmax over a linear
#    function of the one-hot encodings of the previous $K=8$ tokens:
#    $\text{logits}_t = b + \sum_{k=0}^{K-1} W_k\, \mathbf{1}[x_{t-k}]$.
#    It has no hidden layer or nonlinearity before the softmax (multinomial logistic regression).
# 4. LSTM (main model). Embedding (64) → 2-layer LSTM (192 units, dropout 0.2) →
#    dropout → linear layer to vocabulary logits.
#
# Both neural models minimize next-token cross-entropy with Adam.

class LinearContextLM(nn.Module):
    """Multinomial logistic regression on the one-hot encodings of the last K tokens."""
    def __init__(self, vocab, K):
        super().__init__()
        self.K = K
        self.W = nn.ModuleList([nn.Embedding(vocab, vocab) for _ in range(K)])
        for w in self.W: nn.init.zeros_(w.weight)
        self.b = nn.Parameter(torch.zeros(vocab))

    def forward(self, x):
        T = x.shape[1]
        xp = F.pad(x, (self.K - 1, 0), value=PAD)   # PAD acts as "before start" context
        logits = self.b.expand(x.shape[0], T, -1)
        for k in range(self.K):                     # W_k looks k tokens back
            logits = logits + self.W[k](xp[:, self.K - 1 - k: self.K - 1 - k + T])
        return logits

    def next_logits(self, ctx):                      # ctx: (B, K), most recent token last
        return self.b + sum(self.W[k](ctx[:, -1 - k]) for k in range(self.K))


class LSTMLM(nn.Module):
    def __init__(self, vocab, emb, hidden, layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, state=None, return_state=False):
        h, state = self.lstm(self.emb(x), state)
        logits = self.out(self.drop(h))
        return (logits, state) if return_state else logits


def n_params(m):
    return sum(p.numel() for p in m.parameters())

set_seed(CONFIG["seed"])
lstm = LSTMLM(V, CONFIG["lstm_embed"], CONFIG["lstm_hidden"], CONFIG["lstm_layers"],
              CONFIG["lstm_dropout"]).to(DEVICE)
linear = LinearContextLM(V, CONFIG["linear_context"]).to(DEVICE)
print(f"Vocabulary size: {V}")
print(f"LSTM parameters:   {n_params(lstm):,}")
print(f"Linear parameters: {n_params(linear):,}")
out = lstm(xb)
assert out.shape == (*xb.shape, V)
assert linear(xb).shape == (*xb.shape, V)

# ============================================================================================
# 10. Training procedure
# ============================================================================================
#
# Adam, gradient clipping at 1.0, batch size 64. After every epoch the model is evaluated in
# eval mode (no dropout) on the train, validation and test sets. Early stopping: training stops
# when validation loss has not improved for `patience` epochs, and the checkpoint with the
# lowest validation loss is kept. The test loss is recorded only to draw the curve required
# by the course notes.

@torch.no_grad()
def evaluate(model, seq_list, batch_size=128):
    model.eval()
    tot, n = 0.0, 0
    for x, y in make_batches(seq_list, batch_size, False):
        logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                               ignore_index=PAD, reduction="sum").item()
        n += (y != PAD).sum().item()
    return tot / n


def train_model(model, name, lr, max_epochs, patience):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(CONFIG["seed"] + 2)
    hist, best, best_ep, wait = [], float("inf"), 0, 0
    ckpt = f"{OUT}/{name}_best.pt"
    for ep in range(1, max_epochs + 1):
        model.train(); t0 = time.time(); run, cnt = 0.0, 0
        for x, y in make_batches(split_seqs["train"], CONFIG["batch_size"], True, rng):
            opt.zero_grad()
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), ignore_index=PAD)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
            k = (y != PAD).sum().item(); run += loss.item() * k; cnt += k
        row = dict(epoch=ep, train_running=run / cnt,
                   train=evaluate(model, split_seqs["train"]),
                   val=evaluate(model, split_seqs["val"]),
                   test=evaluate(model, split_seqs["test"]), sec=time.time() - t0)
        hist.append(row)
        flag = ""
        if row["val"] < best - 1e-4:
            best, best_ep, wait = row["val"], ep, 0
            torch.save(dict(state_dict=model.state_dict(), config=CONFIG, vocab=ITOS,
                            epoch=ep), ckpt)
            flag = "  *saved"
        else:
            wait += 1
        print(f"{name} ep {ep:2d}  train {row['train']:.4f}  val {row['val']:.4f}  "
              f"test {row['test']:.4f}  ({row['sec']:.0f}s){flag}")
        if wait >= patience:
            print(f"Early stop: no validation improvement for {patience} epochs.")
            break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["state_dict"])
    print(f"{name}: best epoch {best_ep}, val loss {best:.4f}")
    return pd.DataFrame(hist), best_ep

set_seed(CONFIG["seed"])
hist_lin, best_ep_lin = train_model(linear, "linear", CONFIG["linear_lr"],
                                    CONFIG["linear_max_epochs"], CONFIG["patience"])

set_seed(CONFIG["seed"])
hist_lstm, best_ep_lstm = train_model(lstm, "lstm", CONFIG["lstm_lr"],
                                      CONFIG["lstm_max_epochs"], CONFIG["patience"])
hist_lin.to_csv(f"{OUT}/history_linear.csv", index=False)
hist_lstm.to_csv(f"{OUT}/history_lstm.csv", index=False)

# --- Count-based baselines (no gradient training) ---

def count_model(order, alpha):
    C = np.zeros((V, V)) if order == 2 else np.zeros(V)
    for s in split_seqs["train"]:
        for a, b in zip(s[:-1], s[1:]):
            if order == 2: C[a, b] += 1
            else: C[b] += 1
    C = C + alpha
    if order == 2:
        C[:, PAD] = 0; C[:, START] = 0          # never targets
        return C / C.sum(1, keepdims=True)
    C[PAD] = 0; C[START] = 0
    return C / C.sum()

def count_nll(P, order, seq_list):
    tot, n = 0.0, 0
    for s in seq_list:
        for a, b in zip(s[:-1], s[1:]):
            tot -= math.log(P[a, b] if order == 2 else P[b]); n += 1
    return tot / n

alphas = [1e-3, 1e-2, 1e-1, 1.0]
best_alpha = {o: min(alphas, key=lambda a: count_nll(count_model(o, a), o, split_seqs["val"]))
              for o in (1, 2)}
unigram, bigram = count_model(1, best_alpha[1]), count_model(2, best_alpha[2])
print("add-alpha chosen on validation:", best_alpha)

# ============================================================================================
# 11. Training curves (train and test, as required)
# ============================================================================================

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for a, h, name, be in [(ax[0], hist_lin, "Linear baseline", best_ep_lin),
                       (ax[1], hist_lstm, "LSTM", best_ep_lstm)]:
    a.plot(h.epoch, h.train, "-o", ms=3, color="#DD8452", label="train")
    a.plot(h.epoch, h.val, "--", color="#999999", label="validation")
    a.plot(h.epoch, h.test, "-s", ms=3, color="#4C72B0", label="test")
    a.axvline(be, color="k", lw=0.8, ls=":", label=f"selected (ep {be})")
    a.set(title=name, xlabel="epoch"); a.grid(alpha=0.3)
ax[0].set_ylabel("cross-entropy (nats / token)"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{FIG}/training_curves.png", dpi=200); plt.close()

# Compact single-column version of the same curves for the IEEE report
fig, a = plt.subplots(figsize=(3.5, 2.3))
for h, name, c in [(hist_lin, "Linear", "#55A868"), (hist_lstm, "LSTM", "#4C72B0")]:
    a.plot(h.epoch, h.train, "-", color=c, lw=1.4, label=f"{name} train")
    a.plot(h.epoch, h.test, "--", color=c, lw=1.4, label=f"{name} test")
a.set_xlabel("epoch", fontsize=8); a.set_ylabel("cross-entropy (nats/token)", fontsize=8)
a.tick_params(labelsize=7); a.grid(alpha=0.3); a.legend(fontsize=7, frameon=False)
plt.tight_layout(pad=0.3); plt.savefig(f"{FIG}/training_curves_report.pdf"); plt.close()

# ============================================================================================
# 12. Test-set evaluation (next-token prediction)
# ============================================================================================
# Perplexity = exp(mean cross-entropy per predicted token), over the same test tokens
# (including `END`) for every model.

test_ce = {
    "Unigram (reference)": count_nll(unigram, 1, split_seqs["test"]),
    "Bigram Markov": count_nll(bigram, 2, split_seqs["test"]),
    f"Linear (K={CONFIG['linear_context']})": evaluate(linear, split_seqs["test"]),
    "LSTM": evaluate(lstm, split_seqs["test"]),
}
params = {"Unigram (reference)": V - 2, "Bigram Markov": (V - 2) * V,
          f"Linear (K={CONFIG['linear_context']})": n_params(linear), "LSTM": n_params(lstm)}
res = pd.DataFrame({"params": params, "test CE (nats)": test_ce})
res["test perplexity"] = np.exp(res["test CE (nats)"])
print(res.round(4).to_string())
res.to_csv(f"{OUT}/test_results.csv")

# Lower cross-entropy means the model assigns higher probability to unseen pieces from the
# *same generator*. It is not a measure of how good the music sounds.

# ============================================================================================
# 13. Piano sequence generation
# ============================================================================================
#
# Sampling: ancestral sampling from the softmax with temperature T (no top-k / nucleus
# truncation), starting from `START`, up to 400 tokens. `PAD` and `START` are masked out
# because they never occur as targets. No grammar constraints are applied, so validity
# metrics measure what each model learned. Sampling uses a fixed `torch.Generator` seed.
#
# To show that generation works from a saved model, the LSTM is re-created from its
# checkpoint file before sampling.

ck = torch.load(f"{OUT}/lstm_best.pt", map_location=DEVICE)
lstm_reloaded = LSTMLM(V, ck["config"]["lstm_embed"], ck["config"]["lstm_hidden"],
                       ck["config"]["lstm_layers"], ck["config"]["lstm_dropout"]).to(DEVICE)
lstm_reloaded.load_state_dict(ck["state_dict"]); lstm_reloaded.eval()
assert ck["vocab"] == ITOS
assert abs(evaluate(lstm_reloaded, split_seqs["test"]) - test_ce["LSTM"]) < 1e-6
print(f"Reloaded LSTM checkpoint from epoch {ck['epoch']}")

BANNED = torch.tensor([PAD, START], device=DEVICE)

@torch.no_grad()
def sample_lstm(model, n, T, seed, max_len=CONFIG["gen_max_len"]):
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    x = torch.full((n, 1), START, dtype=torch.long, device=DEVICE)
    state, seqs_out = None, [[START] for _ in range(n)]
    done = torch.zeros(n, dtype=torch.bool)
    for _ in range(max_len):
        logits, state = model(x, state, return_state=True)
        logits = logits[:, -1] / T
        logits[:, BANNED] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=g).squeeze(1)
        for i in range(n):
            if not done[i]:
                seqs_out[i].append(int(nxt[i]))
        done |= nxt == END
        if done.all(): break
        x = nxt.view(n, 1).to(DEVICE)
    return seqs_out

@torch.no_grad()
def sample_linear(model, n, T, seed, max_len=CONFIG["gen_max_len"]):
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.eval(); K = model.K
    ctx = torch.full((n, K), PAD, dtype=torch.long, device=DEVICE); ctx[:, -1] = START
    seqs_out, done = [[START] for _ in range(n)], torch.zeros(n, dtype=torch.bool)
    for _ in range(max_len):
        logits = model.next_logits(ctx) / T
        logits[:, BANNED] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=g).squeeze(1)
        for i in range(n):
            if not done[i]: seqs_out[i].append(int(nxt[i]))
        done |= nxt == END
        if done.all(): break
        ctx = torch.cat([ctx[:, 1:], nxt.view(n, 1).to(DEVICE)], 1)
    return seqs_out

def sample_bigram(P, n, T, seed, max_len=CONFIG["gen_max_len"]):
    rng = np.random.default_rng(seed); out = []
    logP = np.log(np.maximum(P, 1e-300)) / T
    for _ in range(n):
        s = [START]
        for _ in range(max_len):
            lp = logP[s[-1]].copy(); lp[[PAD, START]] = -np.inf
            p = np.exp(lp - lp.max()); p /= p.sum()
            s.append(int(rng.choice(V, p=p)))
            if s[-1] == END: break
        out.append(s)
    return out

N, T1 = CONFIG["gen_n_samples"], CONFIG["primary_temperature"]
t0 = time.time()
samples = {
    "Bigram Markov": sample_bigram(bigram, N, T1, seed=11),
    f"Linear (K={CONFIG['linear_context']})": sample_linear(linear, N, T1, seed=12),
}
for T in CONFIG["temperatures"]:
    samples[f"LSTM (T={T})"] = sample_lstm(lstm_reloaded, N, T, seed=13)
print(f"Sampled {N} pieces per setting in {time.time() - t0:.0f}s")

# ============================================================================================
# 14. Generated-sample analysis
# ============================================================================================
#
# | Metric | What it shows | What it does not show |
# |---|---|---|
# | Token validity | share of generated tokens that obey `START (NOTE DUR \| REST)* END` | musical sense |
# | Sequence validity | share of samples with zero grammar errors that end with `END` | |
# | Completion rate | share of samples that emit `END` within 400 tokens | |
# | Bar-line crossing | share of notes/rests that cross a bar line (0 in the data by construction) | whether syncopation would sound good |
# | Bar fit | share of samples whose total length is a whole number of bars | |
# | Scale adherence | share of notes inside the best-fitting major or minor scale | harmonic quality |
# | Tonal closure | share of samples whose final note is the tonic of a scale containing ≥90% of the notes | phrase-level coherence |
# | TV distances | total-variation distance of pitch-class, interval and duration histograms to the test set (0 = identical) | anything sequential |
# | Distinct-4 / repetition | diversity across samples / within a sample | novelty in any musical sense |
# | Exact copies, 8-gram overlap | memorization of training pieces | |
#
# Pitch range validity is guaranteed by the vocabulary (only MIDI 48–84 exist), and is
# re-checked on the exported MIDI file.

SCALES = []
for tonic in range(12):
    SCALES.append((tonic, "major", {(tonic + i) % 12 for i in MAJOR}))
    SCALES.append((tonic, "minor", {(tonic + i) % 12 for i in MINOR} | {(tonic + 11) % 12}))

train_set = {tuple(s) for s in split_seqs["train"]}
def note_ngrams(ev, n):
    toks = [(p, d) for p, d in ev]
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
train_8grams = {g for ev in split_events["train"] for g in note_ngrams(ev, 8)}

def hist(values, bins):
    c = Counter(values); v = np.array([c[b] for b in bins], float)
    return v / max(v.sum(), 1)

PC_BINS, IV_BINS, DUR_BINS = list(range(12)), list(range(-12, 13)), NOTE_DURS

def piece_stats(ev):
    notes = [p for p, _ in ev if p is not None]
    pos, cross = 0, 0
    for _, d in ev:
        if (pos % BAR) + d > BAR: cross += 1
        pos += d
    pcs = [p % 12 for p in notes]
    adher = max(sum(pc in s for pc in pcs) for _, _, s in SCALES) / len(pcs) if pcs else 0
    closure = False
    if pcs:
        t = pcs[-1]
        for tonic, _, s in SCALES:
            if tonic == t and sum(pc in s for pc in pcs) >= 0.9 * len(pcs):
                closure = True
    return dict(total=pos, cross=cross, n_events=len(ev), adher=adher, closure=closure,
                notes=notes, intervals=[int(np.clip(b - a, -12, 12)) for a, b in zip(notes, notes[1:])],
                durs=[d for p, d in ev if p is not None])

def summarize(seq_list, is_reference=False):
    parsed = [parse_ids(s) for s in seq_list]
    evs = [p[0] for p in parsed if len([e for e in p[0] if e[0] is not None]) >= 2]
    st = [piece_stats(e) for e in evs]
    all_pc = [n % 12 for s in st for n in s["notes"]]
    all_iv = [i for s in st for i in s["intervals"]]
    all_du = [d for s in st for d in s["durs"]]
    # 4-grams of (interval, duration) pairs -> transposition-invariant diversity
    def ivgrams(e):
        nd = [(p, d) for p, d in e if p is not None]
        toks = [(b[0] - a[0], b[1]) for a, b in zip(nd, nd[1:])]
        return [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    g_all = [g for e in evs for g in ivgrams(e)]
    rep = np.mean([1 - len(set(ivgrams(e))) / max(len(ivgrams(e)), 1) for e in evs]) if evs else np.nan
    ov8 = [g in train_8grams for e in evs for g in note_ngrams(e, 8)]
    return dict(
        token_validity=1 - sum(p[1] for p in parsed) / sum(p[2] for p in parsed),
        seq_validity=np.mean([p[1] == 0 and p[3] for p in parsed]),
        completion=np.mean([p[3] for p in parsed]),
        bar_crossing=sum(s["cross"] for s in st) / max(sum(s["n_events"] for s in st), 1),
        bar_fit=np.mean([s["total"] % BAR == 0 for s in st]),
        scale_adherence=np.mean([s["adher"] for s in st]),
        tonal_closure=np.mean([s["closure"] for s in st]),
        mean_notes=np.mean([len(s["notes"]) for s in st]),
        pc_hist=hist(all_pc, PC_BINS), iv_hist=hist(all_iv, IV_BINS), du_hist=hist(all_du, DUR_BINS),
        distinct4=len(set(g_all)) / max(len(g_all), 1),
        repetition4=rep,
        exact_copy=np.mean([tuple(s) in train_set for s in seq_list]) if not is_reference else 0.0,
        overlap8=np.mean(ov8) if ov8 else np.nan,
    )

ref = summarize(split_seqs["test"], is_reference=True)
summ = {"Test set (reference)": ref}
summ.update({k: summarize(v) for k, v in samples.items()})

tv = lambda a, b: 0.5 * np.abs(a - b).sum()
rows = []
for k, s in summ.items():
    rows.append({"model": k, "token valid": s["token_validity"], "seq valid": s["seq_validity"],
                 "completed": s["completion"], "bar-cross": s["bar_crossing"], "bar fit": s["bar_fit"],
                 "scale adh.": s["scale_adherence"], "tonal closure": s["tonal_closure"],
                 "TV pitch-class": tv(s["pc_hist"], ref["pc_hist"]),
                 "TV interval": tv(s["iv_hist"], ref["iv_hist"]),
                 "TV duration": tv(s["du_hist"], ref["du_hist"]),
                 "distinct-4": s["distinct4"], "repeat-4": s["repetition4"],
                 "exact copy": s["exact_copy"], "8-gram in train": s["overlap8"],
                 "notes/piece": s["mean_notes"]})
gen_table = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 250)
print(gen_table.round(3).T.to_string())
gen_table.to_csv(f"{OUT}/generation_metrics.csv")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
show = ["Test set (reference)", "Bigram Markov", f"Linear (K={CONFIG['linear_context']})", f"LSTM (T={T1})"]
w = 0.2
for j, k in enumerate(show):
    ax[0].bar(np.arange(12) + (j - 1.5) * w, summ[k]["pc_hist"], w, label=k)
    ax[1].plot(IV_BINS, summ[k]["iv_hist"], "-o", ms=2, label=k)
    ax[2].bar(np.arange(len(DUR_BINS)) + (j - 1.5) * w, summ[k]["du_hist"], w, label=k)
ax[0].set(title="Pitch classes", xticks=range(12),
          xticklabels=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
ax[1].set(title="Melodic intervals (semitones)")
ax[2].set(title="Note durations (16ths)", xticks=range(len(DUR_BINS)), xticklabels=DUR_BINS)
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/generated_vs_test.png", dpi=150); plt.close()

# ============================================================================================
# 15. MIDI and WAV export
# ============================================================================================
#
# Primary piece selection rule (objective, decided before listening). Sample LSTM
# candidates at T = 1.0 with generator seed 2026 and take the first candidate that:
# (1) has no grammar errors and ends with `END`, (2) has no bar-line crossings and a whole
# number of bars, (3) lasts 30–60 s at 90 BPM (i.e. 12 or more bars, long enough to listen to
# phrase structure), (4) is not an exact copy of a training piece.
#
# MIDI rendering (documented post-processing). Acoustic Grand Piano (program 0),
# 90 BPM, velocity 80, each note sounds for 95% of its written length so that repeated
# pitches are re-articulated. No notes are added, removed or changed.

SIXTEENTH = 60.0 / CONFIG["tempo_bpm"] / 4

def events_to_midi(events, path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=CONFIG["tempo_bpm"])
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    inst = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    t = 0.0
    for p, d in events:
        if p is not None:
            inst.notes.append(pretty_midi.Note(CONFIG["velocity"], p, t, t + 0.95 * d * SIXTEENTH))
        t += d * SIXTEENTH
    pm.instruments.append(inst)
    pm.write(path)
    return t

def passes_rule(ids):
    ev, bad, _, ended = parse_ids(ids)
    if bad or not ended or len(ev) == 0: return False
    st = piece_stats(ev)
    dur_s = st["total"] * SIXTEENTH
    return (st["cross"] == 0 and st["total"] % BAR == 0 and 30 <= dur_s <= 60
            and tuple(ids) not in train_set)

cands = []
for b in range(10):                       # batches of 50 candidates, seeds 2026, 2027, ...
    cands += sample_lstm(lstm_reloaded, 50, CONFIG["primary_temperature"], seed=2026 + b)
    ok = [i for i, c in enumerate(cands) if passes_rule(c)]
    if ok: break
assert ok, "no candidate passed the selection rule; train longer (LONG_RUN=True)"
print(f"{len(ok)}/{len(cands)} candidates pass the rule; selected candidate index {ok[0]}")
primary_ids = cands[ok[0]]
primary_events = parse_ids(primary_ids)[0]
primary_path = f"{OUT}/generated_piano.mid"
dur = events_to_midi(primary_events, primary_path)
print(f"Wrote {primary_path}: {len(primary_events)} events, {dur:.1f} s")
print(pretty(primary_events, 40))

# alternate temperatures: the FIRST sample of each run, exported as-is (lenient parse)
for T in CONFIG["temperatures"]:
    ids = samples[f"LSTM (T={T})"][0]
    ev, bad, n, ended = parse_ids(ids)
    p = f"{OUT}/generated_piano_T{T}.mid"
    d = events_to_midi(ev, p)
    print(f"{p}: {d:.1f} s, grammar errors={bad}, ended={ended}")

def verify_midi(path):
    pm = pretty_midi.PrettyMIDI(path)
    notes = [n for inst in pm.instruments for n in inst.notes]
    issues = []
    if any(n.end <= n.start for n in notes): issues.append("non-positive note length")
    if any(not (CONFIG["pitch_min"] <= n.pitch <= CONFIG["pitch_max"]) for n in notes):
        issues.append("pitch out of range")
    by_pitch = {}
    for n in sorted(notes, key=lambda n: n.start):
        if n.pitch in by_pitch and n.start < by_pitch[n.pitch] - 1e-6:
            issues.append(f"overlapping/stuck note {n.pitch}")
        by_pitch[n.pitch] = n.end
    if any(inst.program != 0 or inst.is_drum for inst in pm.instruments):
        issues.append("non-piano instrument")
    _, tempi = pm.get_tempo_changes()
    return dict(file=path, duration_s=round(pm.get_end_time(), 2), notes=len(notes),
                min_pitch=min(n.pitch for n in notes), max_pitch=max(n.pitch for n in notes),
                tempo_bpm=[round(t, 2) for t in tempi], issues=issues or "none")

check = verify_midi(primary_path)
print(json.dumps(check, indent=1))
assert check["issues"] == "none"

# Optional WAV rendering with FluidSynth + any General MIDI soundfont.
SF2_CANDIDATES = [os.environ.get("SOUNDFONT", ""), "/usr/share/sounds/sf2/FluidR3_GM.sf2",
                  "/usr/share/sounds/sf2/default-GM.sf2", "/usr/share/sounds/sf2/TimGM6mb.sf2",
                  "/usr/share/soundfonts/default.sf2"]
sf2 = next((p for p in SF2_CANDIDATES if p and os.path.exists(p)), None)
wav_path = f"{OUT}/generated_piano.wav"
if shutil.which("fluidsynth") and sf2:
    subprocess.run(["fluidsynth", "-ni", "-g", "0.8", "-F", wav_path, "-r", "44100", sf2,
                    primary_path], check=True, capture_output=True)
    print(f"Rendered {wav_path} with soundfont {os.path.basename(os.path.realpath(sf2))}")
else:
    print("FluidSynth or a soundfont was not found; WAV not created. To render it:\n"
          "  sudo apt-get install fluidsynth fluid-soundfont-gm   (or: brew install fluid-synth)\n"
          f"  fluidsynth -ni -F {wav_path} -r 44100 /path/to/GM.sf2 {primary_path}")

fig, ax = plt.subplots(figsize=(12, 2.8))
t = 0.0
for p, d in primary_events:
    if p is not None:
        ax.add_patch(plt.Rectangle((t, p - 0.4), d * SIXTEENTH * 0.95, 0.8, color="#4C72B0"))
    t += d * SIXTEENTH
for b in np.arange(0, t + 1e-9, BAR * SIXTEENTH):
    ax.axvline(b, color="#cccccc", lw=0.6, zorder=0)
notes_p = [p for p, _ in primary_events if p is not None]
ax.set(xlim=(0, t), ylim=(min(notes_p) - 2, max(notes_p) + 2), xlabel="time (s)",
       ylabel="MIDI pitch", title="generated_piano.mid (bar lines in grey)")
plt.tight_layout(); plt.savefig(f"{FIG}/generated_pianoroll.png", dpi=150); plt.close()

# ============================================================================================
# 16. Limitations and possible improvements
# ============================================================================================
#
# - Synthetic data. The model learns a hand-written rule set. Good scores here mean it
#   matches those rules; they do not transfer to real piano repertoire.
# - Monophony and fixed dynamics. No chords, pedal or velocity are modelled; real piano
#   music would need a polyphonic event representation (e.g. note-on/note-off/time-shift).
# - Metrics. All metrics are objective proxies. No listening study was run, so no claim is
#   made about how the music sounds.
# - Single run. Each model was trained once with one seed; differences smaller than
#   run-to-run variation have not been measured.
# - Long-range form. The generator imposes phrase structure (motif return, cadences) that a
#   next-token model can only pick up implicitly; tonal closure measures this only roughly.
#
# Possible extensions: train on a real symbolic corpus (e.g. MAESTRO MIDI) as a comparison,
# add polyphony, add a small Transformer, and run several seeds per model.

# ============================================================================================
# 17. Artifact export and reproduction
# ============================================================================================

summary = dict(
    team=["Sai Abhishek Bhattiprolu", "Shivam Sharma", "Vansh Bharat Gala"],
    config=CONFIG, vocab_size=V,
    data_file=DATA_FILE, n_pieces=len(seqs),
    split_pieces={k: len(v) for k, v in split_seqs.items()},
    split_tokens={k: int(sum(len(x) for x in v)) for k, v in split_seqs.items()},
    params=dict(lstm=n_params(lstm), linear=n_params(linear), bigram=(V - 2) * V),
    best_epoch=dict(lstm=best_ep_lstm, linear=best_ep_lin),
    epochs_run=dict(lstm=len(hist_lstm), linear=len(hist_lin)),
    sec_per_epoch=dict(lstm=float(hist_lstm.sec.mean()), linear=float(hist_lin.sec.mean())),
    best_alpha=best_alpha,
    test_ce={k: float(v) for k, v in test_ce.items()},
    test_ppl={k: float(np.exp(v)) for k, v in test_ce.items()},
    generation=gen_table.astype(float).round(4).to_dict(orient="index"),
    primary=dict(candidate_index=ok[0], pass_count=len(ok), n_candidates=len(cands), events=len(primary_events), **check),
    device=str(DEVICE), torch=torch.__version__,
)
with open(f"{OUT}/results_summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=str)
for fn in sorted(os.listdir(OUT)):
    print(fn)
print("\nDone. Primary piece:", primary_path)

# To reproduce: install the packages in Section 3 and run
# `python APM527_Piano_Music_Generation.py` (default configuration). Data generation, splits, initialization, batching order and
# sampling are all seeded; on the same hardware and library versions the outputs repeat
# exactly. Different PyTorch versions or GPU kernels can change floating-point results slightly,
# which may change sampled pieces while leaving the summary metrics close.
