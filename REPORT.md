# Task 4 — improving the leaf classifier

## Summary

The best result in this experiment was `group_plus_archive_all`: a fine-tuned MobileNetV3-small trained on our photographs and last year's archive, with colour augmentation, AdamW, label smoothing and a cosine learning-rate schedule for 30 epochs.

It reached **74.6%** on the camera-like validation set and **90.8%** on the archive validation set. The Task 3d baseline reached 37.1% and 39.4% on those sets, respectively. This makes `group_plus_archive_all` the strongest Task 4 candidate for the phone. **It is not the model currently running on the Pages site:** `leaf_model.onnx` is still the Task 2 export. The Task 4 results file contains metrics and validation histories, not saved model weights, so the selected variant still has to be trained/exported before it can replace the live model. The camera-like set is also only a proxy, because it is made from the same 29 validation photographs.

## Experimental setup

I used the Task 3d network as the baseline: LeafNetBN, geometric augmentation, Adam with a learning rate of 0.001, 15 epochs and a batch size of 32. It was trained on the group's 107 training photographs.

I compared 15 variants in 33 runs. The important variants were repeated with seeds 0, 1 and 2. The remaining variants were run once, so their results are useful for deciding what to test next, but they are not strong evidence on their own.

| Dataset | What it contains | Number of images |
|---|---|---:|
| `val` | Our original validation photographs | 29 |
| `camera` | Eight fixed-seed crops/rotations/colour variations of each of the 29 validation photographs | 232 |
| `archive` | Last year's validation photographs, taken by other students using other phones | 120 |

The archive training set contains 700 images. Together with our 107 training images, the combined training set contains 807 images. The archive data came from [the course archive ZIP](https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip) (SHA-256: `f7bbaa9a…cf53`). The camera-like set was generated once with a fixed seed, before the model comparisons, so every variant was evaluated on the same images.

**The results are noisy, especially on the original validation set.** Across the nine variants repeated with three seeds, the `val` accuracy spread ranged from 6.9 to 20.7 percentage points; the largest spread was for `leafnetbn_dropout`. On the 232-image camera-like set, the seed spread ranged from 2.2 to 9.5 percentage points. One image changes the `val` score by about 3.4 points and the camera-like score by about 0.43 points. I therefore treat small changes as inconclusive, and the six single-seed variants as preliminary.

All runs were measured on a CPU using 8 threads. The full set of 33 runs took about an hour. Individual runs took between roughly 17 and 596 seconds, depending mainly on the size of the training set and the number of epochs.

## Results

The table shows the mean of the runs for each variant. Values in brackets are the minimum and maximum across seeds. `Δ camera` is the difference from the baseline on the camera-like set.

| Variant | Change | Train images | Trainable parameters | Val 29 (%) | Camera 232 (%) | Archive 120 (%) | Δ camera (pp) | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | Task 3d LeafNetBN | 107 | 208,644 | 62.1 (55–72) | 37.1 (34–39) | 39.4 (34–43) | — | 3 |
| `no_augmentation` | No augmentation | 107 | 208,644 | 67.8 (66–72) | 35.9 (32–38) | 38.1 (34–41) | -1.1 | 3 |
| `colorjitter` | Adds colour augmentation | 107 | 208,644 | 69.0 (66–72) | 51.3 (48–56) | 45.3 (43–47) | +14.2 | 3 |
| `leafnetbn_gap` | Global average pooling | 107 | 65,284 | 43.7 (34–48) | 35.1 (31–37) | 35.3 (31–39) | -2.0 | 3 |
| `leafnetbn_dropout` | Dropout 0.3 in the head | 107 | 208,644 | 57.5 (45–66) | 36.2 (34–38) | 41.7 (39–44) | -0.9 | 3 |
| `mobilenet_frozen` | MobileNetV3-small, only the final layer trained | 107 | 594,948 | 72.4 | 56.9 | 30.8 | +19.8 | 1 |
| `mobilenet_finetuned` | Fine-tuned MobileNetV3-small | 107 | 1,521,956 | 64.4 (59–69) | 55.0 (51–57) | 43.6 (42–45) | +18.0 | 3 |
| `mobilenet_dropout04` | Dropout 0.4 in the MobileNet head | 107 | 1,521,956 | 62.1 | 60.8 | 50.0 | +23.7 | 1 |
| `mobilenet_cos30_color` | Colour augmentation, cosine schedule, 30 epochs | 107 | 1,521,956 | 72.4 (69–76) | 67.4 (66–69) | 58.6 (56–61) | +30.3 | 3 |
| `mobilenet_adamw` | AdamW, weight decay 0.01 | 107 | 1,521,956 | 62.1 | 51.7 | 40.8 | +14.7 | 1 |
| `mobilenet_labelsmooth` | Label smoothing 0.1 | 107 | 1,521,956 | 65.5 | 56.9 | 46.7 | +19.8 | 1 |
| `mobilenet_best_epoch` | Keeps the best validation epoch | 107 | 1,521,956 | 75.9 | 62.1 | 45.0 | +25.0 | 1 |
| `archive_only` | Trained only on last year's photographs | 700 | 1,521,956 | 65.5 | 53.9 | 90.0 | +16.8 | 1 |
| `group_plus_archive` | Adds the archive training photographs | 807 | 1,521,956 | 79.3 (72–90) | 58.2 (53–62) | 87.5 (83–92) | +21.1 | 3 |
| **`group_plus_archive_all`** | Combined data, colour augmentation, AdamW, label smoothing, cosine schedule and 30 epochs | 807 | 1,521,956 | **79.3 (76–83)** | **74.6 (74–76)** | **90.8 (88–93)** | **+37.5** | 3 |

## What made the biggest difference

**Transfer learning helped.** Fine-tuning MobileNetV3-small improved the camera-like accuracy from 37.1% to 55.0%, an increase of **18.0 percentage points** (rounded to one decimal place, as in the results table). This was a much better starting point than training the small network from scratch.

**Colour augmentation also helped.** Adding ColorJitter to the small network increased camera-like accuracy from 37.1% to 51.3%. This suggests that changes in lighting and colour are important for this task. Removing the geometric augmentation, on the other hand, did not make the model better on the camera-like or archive sets.

**The longer MobileNet training recipe performed well.** The MobileNet variant with colour augmentation, a cosine schedule and 30 epochs reached 67.4% on the camera-like set. It was the strongest result before the combined-data experiment was completed.

**The combination using the archive data was the best overall.** The `group_plus_archive` variant, trained for 15 epochs, reached 58.2% on the camera-like set. The final combination reached 74.6%, a difference of 16.4 percentage points. It also achieved the highest archive accuracy in the table.

This last comparison changes several things at once: colour augmentation, AdamW, label smoothing, the cosine schedule and the number of epochs. It shows that the combination works well, but it does not tell us which of those changes was responsible for the improvement. A useful follow-up would be to train on the combined dataset with the longer schedule and colour augmentation, but without AdamW and label smoothing.

## Changes that did not help much

- **Global average pooling** reduced the number of trainable parameters, but also reduced camera-like accuracy to 35.1%.
- **Dropout in the small network** did not improve the result. The change was smaller than the variation between seeds.
- **Removing augmentation** slightly reduced camera-like accuracy. The higher score on the small original validation set is not enough to call it an improvement.
- **AdamW on its own** scored 51.7% on the camera-like set, below the fine-tuned MobileNet result. This was a single run.
- **Label smoothing on its own** reached 56.9%. This was also a single run, so the small difference is not conclusive.
- **Freezing the MobileNet feature extractor** looked good on our 29 validation images, but its archive score was only 30.8%, the lowest in the table. It did not generalise well to the other photo session.

Dropout 0.4 in the MobileNet head and keeping the best epoch both looked promising, but each was tested only once. I would repeat them with more seeds before drawing a firm conclusion.

## Per-class results and common mistakes

For the selected `group_plus_archive_all` variant, the following values are averages across its three seeds.

| Class | Camera-like accuracy (%) | Archive accuracy (%) |
|---|---:|---:|
| apple | 70.8 | 85.6 |
| cherry | 57.3 | 82.2 |
| chestnut | 61.5 | 100.0 |
| maple | 99.1 | 95.6 |

The most frequent error was confusing apple and cherry. Across the three camera-like runs, cherry was predicted as apple 79 times and apple as cherry 55 times. On the archive set, these two directions accounted for 29 mistakes across the three runs. Maple was classified very reliably, while cherry was the weakest class on the camera-like images.

The original validation set is small and not perfectly balanced: it contains only four chestnut images. One incorrectly classified image therefore changes the accuracy noticeably. I would not interpret small differences in per-class accuracy as reliable without more photographs.

## Limitations

There are several reasons not to treat these results as a final estimate of real phone-camera performance:

1. The original validation set contains only 29 photographs. One image changes its accuracy by about 3.4 percentage points.
2. The camera-like set contains 232 images, but they are generated from the same 29 photographs. The transformations make the test more realistic, but they do not introduce new leaves, backgrounds, phones or lighting conditions.
3. The archive set contains photographs from another group, which makes it useful for testing generalisation. However, the archive training and validation images still come from the same collection and may share characteristics.
4. Six variants were tested with only one seed. Their results should be treated as preliminary.
5. The final combined variant changes several training settings at once. More controlled experiments would be needed to understand the contribution of each setting.

The most useful next step would be to collect a separate test set using the actual phone camera, with different backgrounds and lighting, and keep those photographs out of training.

## Model I would use on the phone

I would choose **`group_plus_archive_all` as the next model to deploy**. It had the best mean camera-like and archive results, and its camera-like scores across the three seeds were 73.7–75.9%. Its MobileNetV3-small architecture is compatible with the browser app's existing input contract: 96 × 96 RGB input, the same channel-wise mean and standard deviation, and four output classes.

However, the live page has **not** been updated to this variant. The current `leaf_model.onnx` is still the Task 2 model. `task4_results.json` stores accuracy, per-class results and training histories, but no `state_dict` or checkpoint; `task4.py` also does not currently save the trained weights. The `group_plus_archive_all` checkpoint therefore cannot be exported from the files currently in the repository. It must be regenerated by rerunning that training variant with the original group and archive training images, saving its weights, and then exporting those weights using the Task 2e ONNX export step.

After export, replace `leaf_model.onnx` and regenerate `check_expected.json` for the new model (the old expected probabilities belong to the Task 2 model). Then run the browser self-check again. The preprocessing and page code should not need to change, but the new model must pass its own updated self-check before it is described as deployed.

## Reproducing the experiment

The experiment runner is `task4.py`, and the recorded results are in `task4_results.json`.

```bash
python task4.py --data <data-folder> --out task4_results.json
```

The data folder should contain `leaves/{train,val}` and, for the archive experiments, `leaves_archive/{train,val}`. The archive ZIP has a top-level folder also named `leaves`, so it should be extracted separately and renamed before running the experiment.

The script supports `--only name` (or a comma-separated list of names) to run selected variants, `--force` to rerun completed experiments, and `--quick` for a two-epoch, one-seed smoke test. The results in this report were measured on CPU with 8 threads. Different hardware or library versions may produce slightly different values.
