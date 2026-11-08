# Requirements Audit — APM 527 Final Project (handwritten "Lecture 9" notes)

Source: `lecture9_APM527.pdf`, 2 pages of handwritten notes. Every page was inspected at
250 dpi. Pages contain no typed text layer, so everything below is a transcription.

## 1. Typed transcription

### Page 1 — "Lecture 9"

**Project:**
- Presentation ~~or report~~ *(the word "report" is struck through with a red X)*
  - ↳ 25 min + 5 min questions
- due date: • presentation final week 05/04–05/08
  - ↳ send email to reserve time slot
- up to 3 per project

**Requirement:**
1. use a data set (downloaded or generated)
2. show training for both train and test sets
   - *(sketch: "loss" on the vertical axis, "epochs" on the horizontal axis; an orange curve
     labelled "train" and a cyan curve labelled "test"; both decrease, and the test curve
     levels off slightly above the train curve)*
3. use at least one "baseline model"
   - ↳ ex: compare the result with a linear model
   - ⚠ *(red warning triangle)* do not just train one deep n.n. (e.g. ResNet, Bert)

**Idea:**
- your research topic
- "applied" → Kaggle competition
  - ↳ image classification, sentiment analysis ...  (↳ cifar, ↳ imdb)
- "Theory": diffusion models, graph conv. network, auto-encoders (↳ generative model)
- "art": style transfer (↳ too simple)

### Page 2 — "Specific examples"
1. recognize music instrument → classification → input: sound wave not images
   *(drawing of a waveform in a box, pink arrow → "piano")*
2. detect multiple objects in an image ↳ Regional-CNN *(drawing of a scene with sun, boat)*
3. 1-pixel attack & art transfer ↳ modify image input → fool n.n. / → looks like painting
4. Denoise with Auto-encoders ↳ CNN
5. Generate music (piano) ↳ RNN
6. ⋮

## 2. Mandatory requirements checklist

| # | Requirement (from PDF) | How this project satisfies it |
|---|---|---|
| M1 | Use a dataset, downloaded **or generated** | Deterministic procedural generator of symbolic piano melodies (no external data) |
| M2 | Show training for **both train and test sets** (loss vs. epochs) | Per-epoch train, validation and test cross-entropy curves for the LSTM and the linear baseline. Test loss is only *monitored*; model selection uses validation |
| M3 | At least one **baseline**, e.g. a **linear model** | (a) linear softmax model on a one-hot context window (the PDF's suggested baseline), (b) bigram Markov chain, plus a unigram reference |
| M4 | Do **not** just train one deep network | Three model families compared on identical splits and metrics |
| M5 | Up to 3 people per project | Team of 3 (Bhattiprolu, Sharma, Gala) |
| M6 | Final deliverable: **presentation**, 25 min + 5 min questions | **Not produced here** — see conflict C1 |
| M7 | Present in final week 05/04–05/08; email to reserve a slot | Action item for the team — see ambiguity U4 |

## 3. Optional suggestions / project ideas in the PDF
- Own research topic; Kaggle-style applied tasks (CIFAR image classification, IMDB sentiment);
  theory-oriented generative models (diffusion, GCN, auto-encoders); style transfer (marked "too simple").
- Specific examples: instrument recognition from sound waves; R-CNN object detection; one-pixel
  attack / art transfer; denoising auto-encoders; **piano music generation with an RNN (example 5)**.
- The chosen topic is example 5 directly, and the suggested architecture (RNN) is used (LSTM).

## 4. Grading criteria, formats, naming, restrictions
- **Grading criteria:** none written in the PDF.
- **File formats / naming conventions:** none specified. Names used here follow the team's own brief.
- **Report format / page limit:** none in the PDF (the report option is crossed out).
- **Model restrictions:** only "do not just train one deep n.n."; RNN suggested for music generation.
- **Data restrictions:** dataset may be downloaded or generated; no other restriction.
- **Library / collaboration restrictions:** none written, other than ≤ 3 people per project.

## 5. Conflicts between the PDF and the team brief
- **C1 — Report vs. presentation.** The PDF strikes out "report", leaving a 25 + 5 minute
  presentation as the graded format. The team brief asks for a two-page IEEE report. The report
  is still produced (the brief asks for it and it does not contradict any PDF requirement), but
  the team **should confirm with the instructor** whether a written report is accepted or whether
  slides are required. Slides are not part of this delivery.
- **C2 — Train/test curves vs. train/val/test protocol.** The brief requires a validation split
  for early stopping; the PDF asks for train **and test** curves. Both are satisfied: curves show
  all three splits, and the test curve plays no role in selecting the checkpoint.

## 6. Uncertain handwritten passages

| Page | Best transcription | Alternative reading | Confidence | Material? |
|---|---|---|---|---|
| 1 | "Presentation or ~~report~~" — report option removed | The red X only corrects a misspelling of "report", so a report is still allowed | Medium | **Yes** — decides the graded format (C1) |
| 1 | "25 mn + 5 mn Questions" = 25 min talk + 5 min Q&A | "25 min total incl. 5 min questions" | High | No (affects talk length only) |
| 1 | "05/04–05/08" (May 4–8, final week) | — | High (legibility) | Possibly: these notes may come from an earlier semester; today's date is Oct 2026, so the team should **verify the current term's dates** |
| 1 | "send email to reserve time slot" | "time slots" | High | No |
| 1 | "up to 3 per project" = group size ≤ 3 | "up to 3 projects" | High | No |
| 1 | "(e.g. ResNet, Bert)" | "ResNet, Best" | Medium-high | No |
| 1 | "ex: compare the result with a linear model" | — | High | Yes — shapes baseline choice (satisfied) |
| 2 | "5) Generate music (piano) ↳ RNN" | "KNN" | Medium-high (context strongly favours RNN) | Low |
| 2 | "→ fool n.n. / → looks like painting" | "fool n.n." could be "fool mn" | Medium | No (not our topic) |

**Safest assumption adopted:** satisfy M1–M4 rigorously, deliver the report the team asked for,
and flag C1 and the dates for confirmation with the instructor.
