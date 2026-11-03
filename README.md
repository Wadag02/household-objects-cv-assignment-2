# Household object recognition with DINOv2

Computer Vision · Assignment 2 · Astana IT University · Nauanov Alikhan, SE-2421.

Classify one photograph as **bottle**, **cup** or **pen**. The local web app shows class scores, individual and rolling mean latency, the inference environment and four nearest original training photographs. The linear head makes the prediction; neighbors explain visual resemblance.

[Report (7 pages)](Assignment2_Report.pdf) · [Narrated demo video](Assignment2_Demo.mp4) · [Original A1 dataset](https://drive.google.com/drive/folders/1_Z2BhUgHSwFBgQngrKp4rNrWXKuuv1iN)

## Run the application

Use 64-bit Python 3.12. The experiment ran on Windows 11, Python 3.12.8, using ONNX Runtime's CPU provider. No GPU or Colab account is needed. From this directory, the Windows commands are:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_model.py
.\.venv\Scripts\python.exe app.py
```

On macOS/Linux use `python3.12 -m venv .venv` and `.venv/bin/python` for the remaining commands. Open **http://127.0.0.1:7860**. Upload a JPEG, PNG or WebP, or try either held-out example. Use Ctrl+C to stop. If the port is busy, run `app.py --port 7861` and open that port instead. Model setup downloads a pinned 88 MB encoder once and verifies its SHA-256; the distribution ZIP already includes this file. The fitted head and retrieval index are included in both copies. No original dataset is needed for inference.

## Results

| Pipeline | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| A1 pixel logistic regression | 0.5750 | 29/60 (48.33%) | 0.4583 |
| A1 ResNet18 | 0.8636 | 50/60 (83.33%) | 0.8264 |
| DINOv2, original training views | 0.9162 | 56/60 (93.33%) | 0.9332 |
| DINOv2, originals and mirrors | 0.9166 | 56/60 (93.33%) | 0.9332 |

Validation selected the mirrored head at update 3. Mirroring did not improve either test metric. Each class contributes 20 test views of **one** unseen object. Scores describe this split, not 60 independent objects.

## Reproduce the experiment

Download and extract the original **Dataset** folder: `Dataset/Bottle/<object>/<photo>.JPG`, likewise `Cup` and `Pen`. Preserve the original object folder and image names. After the application setup, run the following with the virtual environment's Python:

```text
python train.py "path/to/Dataset"
```

The program checks all 420 EXIF-corrected RGB pixel hashes against the saved A1 manifest, checks that objects do not cross splits and matches the 60 A1 test predictions by hash. It trains only on the 300 training views (or 600 including mirrors), chooses checkpoints on the 60 validation views, then scores the fixed variants on the test set. It rewrites the fitted head, training index, metrics and per-image predictions. To regenerate the figures and measure latency on your machine, install `matplotlib==3.11.2` and run `python analyze.py` after training.

- `evidence/` — A1 split, archived predictions, baseline metrics and learning history.
- `results/` — held-out predictions, confusion counts, per-update curves, example galleries and the latency benchmark.
- `models/` — fitted weights, retrieval index and image metadata. `web/gallery/` holds the 300 small gallery images, stripped of EXIF; `web/examples/` contains the two demonstration inputs in the distribution.
- `vision.py`, `train.py`, `app.py` — shared preprocessing, experiment and browser application.

## Provenance and limits

DINOv2: [Oquab et al.](https://arxiv.org/abs/2304.07193), [Meta's implementation](https://github.com/facebookresearch/dinov2), [pinned ONNX export](https://huggingface.co/Xenova/dinov2-small/tree/c2bb04a51fab207c420665f1946016107bffc701). The encoder is distributed under Apache-2.0; attribution and license are in `models/`. The original photographic dataset is from Assignment 1; no new rights to redistribute it are asserted here. Synthetic narration is used in the video. Mirrored training images are augmentations; no AI-generated image is counted as an original.

The app binds to loopback and does not save uploaded files. Neighbors are real images from training, not causal attributions. There is no unknown-object detector. Review backgrounds and access permissions before sharing course materials.

