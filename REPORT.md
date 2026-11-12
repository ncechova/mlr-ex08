# Task 4 — making the leaf classifier better

**Short answer.** The phone should keep a **pretrained MobileNetV3-small fine-tuned
with colour augmentation, a cosine schedule and 30 epochs** — `mobilenet_cos30_color`,
**67.4 %** on a camera-like test set where Task 3d's baseline reaches 37.1 %. It is the
same architecture the live page already runs, so this is a weights-only change: same
ONNX graph, same 96 × 96 preprocessing contract, same size, nothing to change in
`index.html`. Second place goes to the model trained on our photos **plus last year's
archive** — better than it on our own validation set and far better on leaves from a
session we were not in, 9 pp worse on the camera proxy. The row that would combine the
two is the one experiment this sweep did not finish.

The rest of this file is what each change brought, which of those differences are
larger than the noise, what did nothing at all, and where I would not believe the
numbers.

## What I compared, and how

The baseline is **Task 3d unchanged**: my `LeafNetBN`, the geometric augmentation of
3d, Adam with learning rate 10⁻³, 15 epochs, batch 32, the same 107 training photos.
Every row of the table changes **one thing** against it; anything a row does not name
keeps the baseline's value. Every run starts from a fixed seed (the same initial
weights and the same augmentation draws), and the rows that matter are repeated with
seeds 0, 1 and 2.

| | |
|---|---|
| code | [`task4.py`](task4.py) — the same pipeline as the notebook, run as a file; the Task 4 cells 4a–4c of the notebook hold the identical experiment |
| our data | the group ZIP from 2a — 107 training, 29 validation photos |
| last year's data | the archive ZIP [`ex08_leaves_images.zip`](https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip), `sha256 f7bbaa9a…cf53` — 175 training + 30 validation photos per species, 700 + 120 in total, other students, other phones |
| hardware | CPU only, 8 threads; one run 17–301 s, 32 min of CPU for the 30 finished runs |

Neither dataset is in this repository: they are the students' photographs, and the
repository is public.

### Three ways of scoring the same model

| set | what it is | images | one image is worth |
|---|---|---|---|
| `val` | our validation photos, exactly as in Task 2 and 3 | 29 | **3.4 %** |
| `camera` | those 29 photos made camera-like **once, with a fixed seed**: 8 versions each, a random crop of 35–100 % of the area, rotation up to 45°, colours up to ±50 % | 232 | 0.43 % |
| `archive` | last year's validation photos — leaves none of us photographed | 120 | 0.83 % |

The camera-like set is deliberately **stronger than the training augmentation**, so
that no variant can be tuned to a test set it also trains against, and it is built
before any training runs, so every run of the sweep is scored on the identical 232
images.

**The noise, before any result.** On the 29-photo set the *same variant with a
different seed* moved by up to **17 percentage points** (the baseline: 55, 72, 58 %).
One validation image is 3.4 %, so on that set a difference of two or three images is
indistinguishable from re-rolling the initial weights. On the 232-image camera set
the seed spread is 2.6–7.8 pp and one image is 0.43 %. I therefore read the `camera`
and `archive` columns as the results and the `val` column only as continuity with
Tasks 2 and 3 — and I treat any single-seed difference below about 5 pp on the camera
column as not established.

## The table

Mean of the seeds; the bracket is the min–max spread of the runs; `Δ camera` is the
change against the baseline on the column that matters most.

| variant | what one change does | train images | trainable params | val 29 % | camera-like 232 % | archive 120 % | Δ camera vs baseline | seeds | s/run |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | Task 3d: LeafNetBN, geo augmentation, Adam 1e-3, 15 epochs | 107 | 208,644 | 62.1 (55–72) | 37.1 (34–39) | 39.4 (34–43) |  | 3 | 21 |
| `no_augmentation` | − augmentation (only ToTensor + Normalize) | 107 | 208,644 | 67.8 (66–72) | 35.9 (32–38) | 38.1 (34–41) | -1.1 | 3 | 19 |
| `colorjitter` | + ColorJitter to the augmentation | 107 | 208,644 | 69.0 (66–72) | 51.3 (48–56) | 45.3 (43–47) | +14.2 | 3 | 24 |
| `leafnetbn_gap` | + global average pooling instead of flatten | 107 | 65,284 | 43.7 (34–48) | 35.1 (31–37) | 35.3 (31–39) | -2.0 | 3 | 20 |
| `leafnetbn_dropout` | + Dropout(0.3) in the head | 107 | 208,644 | 57.5 (45–66) | 36.2 (34–38) | 41.7 (39–44) | -0.9 | 3 | 24 |
| `mobilenet_frozen` | pretrained MobileNetV3, only the last layer trained | 107 | 594,948 | 72.4 | 56.9 | 30.8 | +19.8 | 1 | 17 |
| `mobilenet_finetuned` | pretrained MobileNetV3, whole net fine-tuned (Task 2) | 107 | 1,521,956 | 64.4 (59–69) | 55.0 (51–57) | 43.6 (42–45) | +18.0 | 3 | 60 |
| `mobilenet_dropout04` | MobileNet + Dropout(0.4) in its head (was 0.2) | 107 | 1,521,956 | 62.1 | 60.8 | 50.0 | +23.7 | 1 | 51 |
| **`mobilenet_cos30_color`** | MobileNet + cosine schedule, 30 epochs, + ColorJitter | 107 | 1,521,956 | **72.4 (69–76)** | **67.4 (66–69)** | **58.6 (56–61)** | **+30.3** | 3 | 72 |
| `mobilenet_adamw` | MobileNet + AdamW (weight decay 0.01) | 107 | 1,521,956 | 62.1 | 51.7 | 40.8 | +14.7 | 1 | 34 |
| `mobilenet_labelsmooth` | MobileNet + label smoothing 0.1 | 107 | 1,521,956 | 65.5 | 56.9 | 46.7 | +19.8 | 1 | 30 |
| `mobilenet_best_epoch` | MobileNet, weights of the best epoch kept | 107 | 1,521,956 | 75.9 | 62.1 | 45.0 | +25.0 | 1 | 28 |
| `archive_only` | MobileNet trained only on last year's photos | 700 | 1,521,956 | 65.5 | 53.9 | 90.0 | +16.8 | 1 | 221 |
| `group_plus_archive` | MobileNet + last year's photos added to training | 807 | 1,521,956 | 79.3 (72–90) | 58.2 (53–62) | 87.5 (83–92) | +21.1 | 3 | 279 |
| `group_plus_archive_all` | the row above **+** colour, AdamW, label smoothing, cosine, 30 epochs | 807 | 1,521,956 | — | — | — | — | 0 | ~500 |

The last row did not finish — the machine ran out of memory on its first of three
runs. It is the row that would settle the open question at the end, so it is listed
here as the experiment still owed rather than quietly dropped.

Accuracy is `100 × correct / images` on each of the three sets; `s/run` is the wall-clock
time of one run on 8 CPU threads, 32 min of CPU for the 30 runs that are in.
[`task4_results.json`](task4_results.json) holds every run, including the per-epoch
validation history and the confusion counts behind the next section.

## What helped

1. **Transfer learning — the largest single factor on the camera set: +18.0 pp**
   (fine-tuned MobileNet over the baseline, three seeds each, whose own spread is
   5.6 pp). The pretrained weights are worth most where the input stops resembling the
   training photos. Note where they are worth *least*: on last year's photos the same
   comparison is only **+4.2 pp**, and `colorjitter` trained from scratch (45.3 %)
   matches the plain fine-tuned MobileNet (43.6 %) there — Task 2's text quotes 56 %
   against 75 % on that data, and on our 29 photos I cannot reproduce a gap of that
   size. What the pretrained network buys is robustness to the camera, not to another
   photo session; the two are different problems and my two test sets separate them.
2. **Colour augmentation: +14.2 pp camera, +5.9 pp archive** on the from-scratch
   network, three seeds, and the spread of the seeds (7.8 pp) is smaller than the
   gain on the camera set. The geometric augmentation alone barely matters at all —
   see *What did not help*.
3. **Longer training with a cosine schedule (30 epochs) on top of colour: 67.4 %
   camera**, the best mean in the table, and the *most stable* row of all (2.6 pp
   spread over three seeds, against 4.7–7.8 pp for the others).
4. **Keeping the weights of the best epoch: +7.1 pp** — and the best epoch was the
   **7th of 15**, i.e. the fine-tuned network was already getting worse on leaves
   while its training loss kept falling. One caveat below: "best" is chosen on the
   29-photo set, so that row's `val` number is optimistic by construction.
5. **Dropout 0.4 instead of 0.2 in MobileNet's head: +5.8 pp camera, +6.4 pp
   archive.** The cheapest change in the table — one line — and it moves both sets in
   the same direction.
6. **Last year's photographs: +14.9 pp on our validation set and +43.9 pp on the
   archive test**, but only **+3.2 pp on the camera set** (`group_plus_archive`
   against `mobilenet_finetuned`, both 15 epochs of the same network — that is the
   like-for-like pair). Seven and a half times more training photos moved the two sets
   that contain their kind of photographs a long way and the camera proxy barely at
   all, which is itself the result: what the archive data teach is *that* photo
   session's conditions, and my camera-like set is built from ours, so it cannot see
   the benefit. The row that would show the benefit — archive data with the recipe of
   row 9 — is the unfinished one.

## What did not help, or hurt

- **Global average pooling instead of flattening: −2.0 pp camera, and 62 → 44 % on
  our validation set.** It also throws away 143 000 of the 208 000 parameters, so the
  saving is real but the loss is bigger. With 96 × 96 leaves the position of the leaf
  in the frame is apparently information the small network uses.
- **Dropout in the small network: within noise** (−0.9 pp camera). The same idea in
  MobileNet's head helped (+5.8 pp); in a network that already underfits 107 photos,
  turning neurons off takes away capacity rather than curbing overfitting.
- **AdamW with weight decay: −3.3 pp camera** (one seed). Weight decay is a
  regulariser for a network that is not overfitting on this data; I would not spend
  another run on it.
- **Label smoothing 0.1: +1.9 pp** — inside the noise, single seed, no evidence.
- **The frozen MobileNet is the trap of the whole table.** Only its last layer
  trained: `val` 72.4 %, a tenth above the baseline and enough to look like a winner in
  Task 3d's own table — and the **lowest `archive` score of every row**, 30.8 %, worse
  than the small network trained from scratch.
  The head alone adapts to our 107 photographs and to nothing else. Judged on `val`
  alone it looks like a keeper — which is exactly how a model that fails the moment it
  leaves our table gets chosen.
- **Removing the augmentation barely changes anything** (−1.1 pp camera, −1.3 pp
  archive; on `val` it is even 5.7 pp *better* than the baseline). Whatever Task 3d's
  augmentation bought, on the sets that resemble a camera it is not measurable here.
  What measurably helps is the **colour** part I added in 4c, not the flips and
  rotations of 3d.

## Where the mistakes are

The dominant confusion changes with the network, not with the species:

- every **pretrained** row mixes up **apple ↔ cherry** (for the plain fine-tune, 24
  of its 120 archive errors),
- every **from-scratch** row mixes up **chestnut ↔ maple** (15–19 of 120).

The per-class column has to be read with the same suspicion as the accuracy. The
camera set holds 64, 64, 32 and 72 images of apple, cherry, chestnut and maple — our
29 validation photos are 8, 8, **4** and 9 of them, so chestnut rests on four
photographs and one chestnut image is already 3 pp of its column. And the *apple*
column alone swings from 8 % (`mobilenet_finetuned`) to 75 % (`mobilenet_frozen`)
between rows that differ in one line. I read the four numbers of a row together; a
single cell of them is not a finding. Of the two candidate models, the one trained on
our photos *and* the archive's is distinctly the more balanced (spread 27 pp across the
four classes against 43 pp, and no class below 47 %), which is a point in its favour
that the accuracy column alone hides.

## Which model I would put on the phone

**`mobilenet_cos30_color`** — the pretrained MobileNetV3-small, our own photos, colour
augmentation, cosine schedule, 30 epochs.

* It is the best on the set that stands for the phone: **67.4 %**, and it is the only
  row whose three runs land within 2.6 pp of each other. The best other row measured
  three times is 58.2 %; the best single-run probe is 62.1 %. Its lead is larger than
  its own noise, which is not true of most of the table.
* It is also the best of every row that **never saw last year's photographs** on the
  archive test (58.6 %), so the gain is not just a better fit to the conditions of our
  own photo session — the failure mode `mobilenet_frozen` shows so sharply.
* Its four per-class numbers (51 / 56 / 62 / 94) contain no collapsed class, though it
  does lean on maple — see the balance point above, which is the one argument the
  other candidate has.
* **It is a weights-only change.** The `leaf_model.onnx` this repository serves is
  already a MobileNetV3-small — 52 convolutions, 1.52 M parameters, 6.3 MB file
  (5.7 MB over the wire), the same `image` → `logits` graph. Exporting this variant
  through 2e produces the same kind of file: same 96 × 96 input, same mean and std,
  same size and speed on the phone, nothing to change in `index.html` or in the QR code.

**What I did not pick.** `group_plus_archive` has the highest validation accuracy in
the table (79.3 %) and is unambiguously the model that generalises to leaves from
another session (87.5 % against 58.6 %) — and 9.2 pp worse on the camera proxy, which
is the number I said I would trust, so I am not picking it. `mobilenet_frozen` is not
a candidate at all: it is the row that shows how a high `val` and a useless phone
classifier live in the same file.

**The rule I applied, so that you can disagree with it:** among the finished rows I
took the best measured on the camera-like set, not the one I would guess wins. The
guess is that archive data *plus* colour plus cosine beats both rows here — that is
exactly the run that did not finish, and I am not going to promote a model on a guess.
It costs one command and about half an hour:

```bash
python task4.py --data <that folder> --only group_plus_archive_all
```

## What I would not believe in this table

- **29 real photographs.** The camera-like set is derived from those same 29, so it
  is correlated with `val` and it **cannot contain anything the photos do not show**
  — no motion blur, no finger across the leaf, no leaf against a sky, no second leaf
  in the frame. It is a better proxy for the camera than a flat scan is, not the
  camera. The honest next step is one real photo session of my own, kept out of
  training entirely.
- **`archive` is easier for the rows that trained on it.** `archive_only` scores 90 %
  on it and `group_plus_archive` ~88 %, because their training photos come from the
  same session as that test set. The comparison that means something is the other
  direction: `archive_only`, which never saw our photos, still only reaches 65.5 % on
  our `val` and 53.9 % on the camera set.
- **Six rows are a single run** (marked in the *seeds* column) and were chosen as
  cheap probes, not as results.
- **The combination rows change more than one thing at once** — I did not separate
  the cosine schedule from the extra epochs, and 30 epochs without colour is not in
  the table.

## How to reproduce

```bash
# data: a folder holding leaves/{train,val} (the group ZIP) and
#       leaves_archive/{train,val} (the archive ZIP, see above — its top folder is
#       also called "leaves", so unpack it somewhere else and rename it)
python task4.py --data <that folder> --out task4_results.json
```

`task4.py` prints one line per run as it goes and writes the JSON after every run, so
an interrupted sweep resumes where it stopped (`--force` reruns). `--only name,…` runs
part of the table, `--quick` does 2 epochs of one seed as a smoke test. The numbers in
this report are from CPU, 8 threads, PyTorch 2.15 / torchvision 0.29 — a different
library version or a GPU moves a few tenths of a percent, which is inside the noise
above.
