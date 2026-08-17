# White Blood Cell (WBC) Analysis & Classification

A classical (non-deep-learning) digital image processing pipeline that segments the **nucleus** of a white blood cell from a microscope image and classifies the cell into one of **five WBC types** — **Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil** — using hand-engineered features (shape, texture, gradient) and a **Support Vector Machine (SVM)** classifier.

----

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Implementation Details (Cell-by-Cell)](#implementation-details-cell-by-cell)
   - [1. Imports & Setup](#1-imports--setup)
   - [2. Image Loading](#2-image-loading)
   - [3. Butterworth High-Pass Filter](#3-butterworth-high-pass-filter)
   - [4. Histogram Equalization](#4-histogram-equalization)
   - [5. Global Thresholding](#5-global-thresholding)
   - [6. Connected Component Analysis (8-connectivity)](#6-connected-component-analysis-8-connectivity)
   - [7. Nucleus Extraction Pipeline](#7-nucleus-extraction-pipeline)
   - [8. Local Binary Pattern (LBP)](#8-local-binary-pattern-lbp)
   - [9. Histogram of Oriented Gradients (HOG)](#9-histogram-of-oriented-gradients-hog)
   - [10. Nucleus Shape Features](#10-nucleus-shape-features)
   - [11. Full Feature Vector Assembly](#11-full-feature-vector-assembly)
   - [12. Dataset Construction](#12-dataset-construction)
   - [13. Model Training (SVM)](#13-model-training-svm)
   - [14. Model Persistence](#14-model-persistence)
   - [15. Model Loading & Inference](#15-model-loading--inference)
   - [16. Test Set Evaluation](#16-test-set-evaluation)
6. [Feature Vector Composition](#feature-vector-composition)
7. [Results](#results)
8. [How to Run](#how-to-run)
9. [Dependencies](#dependencies)
10. [Known Issues & Limitations](#known-issues--limitations)
11. [Possible Future Improvements](#possible-future-improvements)

---

## Project Overview

The goal of the project is to classify blood-smear microscope images of white blood cells into their five clinical subtypes using **classical image processing** rather than a CNN/deep-learning approach. The system:

1. Loads a grayscale WBC microscope image.
2. Sharpens it in the **frequency domain** with a Butterworth high-pass filter, then re-blends and equalizes the histogram.
3. Thresholds and morphologically closes the image to obtain a binary mask.
4. Runs a **from-scratch 8-connected component labeling algorithm** to isolate the connected blob that corresponds to the cell **nucleus** (the region containing the image center).
5. Extracts a **combined feature vector** per image made of:
   - Local Binary Pattern (LBP) texture codes (per-pixel, flattened, from-scratch implementation),
   - Histogram of Oriented Gradients (HOG) descriptor (from-scratch implementation),
   - Nucleus shape descriptors (area, perimeter, circularity, aspect ratio, centroid).
6. Trains an **RBF-kernel SVM** (`sklearn.svm.SVC`) on the extracted features (with `StandardScaler` normalization).
7. Evaluates the model on a held-out validation split and on a separate test set, reporting accuracy, per-class precision/recall/F1, and confusion matrices.
8. Persists the trained model, scaler, and label map to disk (`joblib` + `json`) for later inference without retraining.

A high-level flowchart of the process is included as [`flowchart.png`](flowchart.png).

---

## Repository Structure

```
White Blood Cell Analysis & Classification/
├── main.ipynb        # Main notebook — the entire pipeline (preprocessing, feature
│                              # extraction, training, evaluation) lives here
├── flowchart.png             # Visual flowchart of the processing pipeline
├── wbc_data/                 # Image dataset (JPEG, 575×575, RGB-encoded but read as grayscale)
│   ├── Train/                # 100 images per class — used to train the SVM
│   │   ├── Basophil/
│   │   ├── Eosinophil/
│   │   ├── Lymphocyte/
│   │   ├── Monocyte/
│   │   └── Neutrophil/
│   ├── Train_small/           # 50 images per class — a lighter-weight training subset
│   │   ├── Basophil/
│   │   ├── Eosinophil/
│   │   ├── Lymphocyte/
│   │   ├── Monocyte/
│   │   └── Neutrophil/
│   └── Test/                  # 50 images per class — held-out evaluation set
│       ├── Basophil/
│       ├── Eosinophil/
│       ├── Lymphocyte/
│       ├── Monocyte/
│       └── Neutrophil/
└── README.md                  # This file
```

> **Note:** The notebook does not currently emit any saved model artifacts (`wbc_classifier_model.pkl`, `wbc_scaler.pkl`, `wbc_label_map.json`) into the repository — see [Model Persistence](#14-model-persistence). These files are created only after the training cells are executed locally.

---

## Dataset

- **Location:** `wbc_data/`
- **Classes (5):** `Basophil`, `Eosinophil`, `Lymphocyte`, `Monocyte`, `Neutrophil`
- **Image format:** JPEG, 575×575 pixels, loaded via OpenCV in grayscale mode (`cv2.imread(path, 0)`).
- **Splits:**

| Split        | Images / class | Total images |
|--------------|----------------|---------------|
| `Train`      | 100            | 500           |
| `Train_small`| 50             | 250           |
| `Test`       | 50             | 250           |

- Filenames follow the pattern `<ClassName>_<index>.jpg` (e.g. `Basophil_1.jpg`, `Eosinophil_2.jpg`).
- The dataset appears to be sourced/derived from a public peripheral blood cell / WBC image dataset (5-class subtype scheme), reorganized into per-class folders compatible with `build_dataset()`'s folder-as-label convention.

---

## Pipeline Architecture

```
Raw grayscale image
        │
        ▼
Butterworth High-Pass Filter (frequency domain, FFT)
        │
        ▼
Add sharpened result back to original + Histogram Equalization
        │
        ▼
Global Thresholding (binary mask, threshold = 128)
        │
        ▼
Morphological Closing (cross-shaped 3×3 kernel)
        │
        ▼
Custom 8-Connected Component Labeling  →  isolates the nucleus blob
        │
        ▼
Nucleus mask (background suppressed to white/255)
        │
        ├──► Shape features (area, perimeter, circularity, aspect ratio, centroid)
        │
Resized 128×128 grayscale image
        │
        ├──► Local Binary Pattern (LBP) — flattened per-pixel codes
        └──► Histogram of Oriented Gradients (HOG) descriptor
        │
        ▼
Concatenated feature vector = [LBP | HOG | shape features]
        │
        ▼
StandardScaler normalization
        │
        ▼
SVM (RBF kernel, class_weight='balanced') — training / prediction
        │
        ▼
Predicted WBC class + accuracy / classification report / confusion matrix
```

---

## Implementation Details (Cell-by-Cell)

The notebook is organized into 16 markdown-delimited sections. Each is detailed below.

### 1. Imports & Setup
```python
import cv2, numpy as np, matplotlib.pyplot as plt, copy, os, joblib, json
```
Core stack: **OpenCV** (I/O, morphology, contours), **NumPy** (array/FFT math), **Matplotlib** (visualization), **joblib**/**json** (model & metadata persistence).

### 2. Image Loading
Five sample images (one per class) are loaded from the `Test` folder in grayscale for demonstration/visualization purposes:
```python
I1 = cv2.imread(".../Test/Basophil/Basophil_1.jpg", 0)
I2 = cv2.imread(".../Test/Eosinophil/Eosinophil_2.jpg", 0)
...
```
### 3. Butterworth High-Pass Filter
`butterworth_high_pass(D0, n, rows, cols)` builds a frequency-domain Butterworth high-pass filter mask `H` of shape `(rows, cols)`:
- Builds a distance-from-center grid `D` using `np.meshgrid`.
- Applies the standard Butterworth HPF formula: `H = 1 / (1 + (D0/D)^(2n))`, with a small epsilon added to `D` to avoid division by zero.
- `D0` = cutoff frequency, `n` = filter order.

Used later to sharpen the image before segmentation (frequency-domain **edge enhancement**).

### 4. Histogram Equalization
`Equalization(I1)` — a **from-scratch** implementation of global histogram equalization:
- Computes the 256-bin histogram and its cumulative distribution function (CDF).
- Normalizes and rescales the CDF to `[0, 255]` as a uint8 lookup table.
- Remaps every pixel through the lookup table (`cdf[I1]`), boosting global contrast.

### 5. Global Thresholding
`global_threshold(image, thresh=128)` — thin wrapper around `cv2.threshold` (binary thresholding) used to convert the equalized/sharpened grayscale image into a binary mask.

### 6. Connected Component Analysis (8-connectivity)
`CCA_8(img, I1)` — a **hand-written two-pass connected-component labeling algorithm** (not using `cv2.connectedComponents`):
- **First pass:** scans the binary image row by row; for each foreground pixel (`< 128`), inspects the left, top, top-left, and top-right neighbors (8-connectivity, causal/forward-scan neighborhood). Assigns a new label if no labeled neighbor exists, otherwise assigns the minimum neighboring label and records label **equivalences** in a dictionary of sets.
- **Second pass:** resolves every pixel's label to the minimum label in its equivalence class.
- **Nucleus selection:** the algorithm treats the label found at the image's center pixel `a[x/2, y/2]` as the nucleus. If the center pixel itself is background (255), it scans outward (bottom-right quadrant) to find the first foreground pixel and uses its component label instead.
- **Masking:** all pixels *not* belonging to the identified nucleus component are set to white (255) in the output image, effectively isolating just the nucleus blob against a white background.

This is the core, purpose-built segmentation step that assumes the nucleus is roughly centered in each cropped cell image (a valid assumption for this cropped-cell dataset).

### 7. Nucleus Extraction Pipeline
`extract_nucleus(I1)` chains together the previous building blocks into the full segmentation pipeline:
1. 2D FFT of the image (`np.fft.fft2`) and shift the zero-frequency component to center (`fftshift`).
2. Build a Butterworth HPF with `D0=15`, `n=1` and multiply it into the shifted spectrum (frequency-domain sharpening).
3. Inverse FFT (`ifftshift` → `ifft2`) and take the magnitude to reconstruct a sharpened spatial-domain image, clipped to `uint8`.
4. **Blend** the sharpened image with the original (`img_sharpened + I1`) and apply the from-scratch histogram equalization.
5. Global-threshold the result at 128 to binarize.
6. **Morphological closing** with a 3×3 cross-shaped structuring element (`cv2.MORPH_CLOSE`) to fill small gaps/holes in the mask.
7. Run the custom `CCA_8` labeling to extract the connected nucleus component.

A demonstration cell (Cell 14) visualizes the original vs. extracted-nucleus image side by side using Matplotlib subplots.

### 8. Local Binary Pattern (LBP)
`local_binary_pattern(I1)` — a **from-scratch, per-pixel LBP** implementation:
- Zero-pads the image by 1 pixel on all sides.
- For each pixel, if the center value is already background (255, i.e. part of the nucleus mask's white background), the LBP output is forced to 0 — LBP is effectively computed **only within the nucleus region**.
- Otherwise, examines the 8 surrounding neighbors in clockwise order (`TL, T, TR, R, BR, B, BL, L`), thresholds each against the center pixel (`neighbor >= center → 1 else 0`), and packs the 8 bits into a single value via `conv_binary` (standard LBP binary-to-decimal encoding, MSB-first).
- Produces a full-resolution LBP-coded image, later flattened into the feature vector (i.e. the **entire spatial LBP map**, not a histogram of codes, is used as the feature — a high-dimensional but spatially-informative texture descriptor).

### 9. Histogram of Oriented Gradients (HOG)
`extract_hog(image, cell_size=8, block_size=2, bins=9)` — a **from-scratch HOG descriptor**:
- Computes horizontal/vertical Sobel gradients (`cv2.Sobel`), then gradient magnitude and orientation (unsigned, `[0, 180)` degrees).
- Divides the image into `cell_size × cell_size` (default 8×8) cells and accumulates a 9-bin orientation histogram per cell, weighted by gradient magnitude.
- Groups cells into overlapping `block_size × block_size` (default 2×2) blocks, concatenates their histograms, and L2-normalizes each block (`+1e-6` epsilon for stability).
- Flattens all normalized block vectors into the final HOG feature vector — this is a manual re-implementation of the classic **Dalal & Triggs HOG** algorithm (as used in pedestrian detection), applied here to capture nucleus/cell edge-orientation structure.

### 10. Nucleus Shape Features
`extract_shape_features(nucleus_img)` — classic **contour-based morphometric features** using OpenCV:
- Inverse-thresholds the nucleus mask (foreground = anything below 254) to recover a clean binary silhouette.
- Finds external contours (`cv2.findContours`, `RETR_EXTERNAL`) and selects the largest by area (in case of noise/multiple components).
- Computes:
  - **Area** (`cv2.contourArea`)
  - **Perimeter** (`cv2.arcLength`)
  - **Circularity** = `4π·Area / Perimeter²` (a shape-compactness measure; 1.0 = perfect circle)
  - **Aspect ratio** = bounding-box width / height
  - **Centroid (cx, cy)** from image moments (`cv2.moments`)
- Returns these six values as a flat list (feature vector), or `None` if no contour is found.

### 11. Full Feature Vector Assembly
`extract_all_features(img)` ties everything together for a single input image:
1. Runs `extract_nucleus(img)` to segment the nucleus.
2. Runs `extract_shape_features` on the segmented nucleus.
3. Resizes the *original* grayscale image to 128×128.
4. Computes the flattened LBP map (128×128 = 16,384 values) on the resized image.
5. Computes the HOG descriptor on the same resized image.
6. Concatenates `[LBP flattened | HOG vector | 6 shape features]` via `np.hstack` into one combined feature vector per image.

### 12. Dataset Construction
`build_dataset(path)` walks a directory of class subfolders:
- Treats each subfolder name as a class label, auto-building a `label_map: {class_name → integer index}` in folder-iteration order.
- For every `.png`/`.jpg`/`.tif` file, loads it grayscale, extracts the full feature vector via `extract_all_features`, and appends it (with its label) to `X`/`y` lists.
- Returns `X` (feature matrix), `y` (integer labels), and `label_map`.

### 13. Model Training (SVM)
- Builds the training feature matrix from `wbc_data/Train` (500 images) via `build_dataset`.
- **Standardizes** features with `sklearn.preprocessing.StandardScaler` (zero mean, unit variance).
- Splits into train/validation (80/20, `random_state=42`) via `train_test_split`.
- Trains an **`SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')`** — the `balanced` class weighting compensates for any class imbalance in the training split.
- Evaluates on the validation split: prints `accuracy_score`, a full `classification_report` (precision/recall/F1 per class), and renders a confusion-matrix heatmap with Seaborn.

### 14. Model Persistence
```python
joblib.dump(model, 'wbc_classifier_model.pkl')
joblib.dump(scaler, 'wbc_scaler.pkl')
json.dump(label_map, open('wbc_label_map.json', 'w'))
```
Saves the trained SVM, the fitted `StandardScaler`, and the class label mapping to disk so inference can be done later without retraining.

### 15. Model Loading & Inference
Reloads the three artifacts above and builds `inv_label_map` (index → class name) for turning predicted integer labels back into human-readable class names.

### 16. Test Set Evaluation
- Runs the **same** `extract_all_features` pipeline over every image in `wbc_data/Test` (250 images, using the *true* folder-derived labels via the already-loaded `label_map`).
- Scales features with the *already-fitted* `scaler` (`scaler.transform`, not `fit_transform` — correct practice, no data leakage).
- Predicts with the loaded model and reports final held-out test accuracy, classification report, and confusion matrix.

---

## Feature Vector Composition

For a 128×128 resized image, the final per-image feature vector has (up to):

| Component            | Source function              | Dimensionality        |
|-----------------------|-------------------------------|------------------------|
| LBP map (flattened)   | `local_binary_pattern`        | 128 × 128 = 16,384     |
| HOG descriptor        | `extract_hog` (8px cells, 2×2 blocks, 9 bins) | 15 × 15 cells → 14×14 blocks × 4 cells/block × 9 bins = 7,056 |
| Shape features         | `extract_shape_features`      | 6 (area, perimeter, circularity, aspect ratio, centroid x/y) |
| **Total**             |                                | **≈ 23,446**            |

This is a very high-dimensional, largely texture-dominated feature space (LBP contributes ~70% of the dimensions) fed into an RBF-SVM after standardization.

---

## Results

### Validation split (80/20 split of `Train`, 100 samples)
- **Accuracy: 0.55**

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Basophil    | 0.96      | 0.86   | 0.91     | 28      |
| Eosinophil  | 0.27      | 0.57   | 0.36     | 14      |
| Lymphocyte  | 0.50      | 0.90   | 0.64     | 10      |
| Monocyte    | 0.36      | 0.21   | 0.26     | 24      |
| Neutrophil  | 0.69      | 0.38   | 0.49     | 24      |
| **Macro avg** | 0.56    | 0.58   | 0.53     | 100     |
| **Weighted avg** | 0.61 | 0.55   | 0.55     | 100     |

### Held-out test set (`wbc_data/Test`, 250 samples)
- **Accuracy: 0.66**

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Basophil    | 0.96      | 0.94   | 0.95     | 50      |
| Eosinophil  | 0.45      | 0.68   | 0.54     | 50      |
| Lymphocyte  | 0.89      | 0.94   | 0.91     | 50      |
| Monocyte    | 0.50      | 0.32   | 0.39     | 50      |
| Neutrophil  | 0.55      | 0.44   | 0.49     | 50      |
| **Macro avg** | 0.67    | 0.66   | 0.66     | 250     |
| **Weighted avg** | 0.67 | 0.66  | 0.66     | 250     |

**Observations:**
- **Basophil** and **Lymphocyte** are classified very reliably (F1 ≥ 0.91 on test) — these classes likely have the most visually distinctive nucleus shape/texture.
- **Monocyte** is the weakest class (F1 = 0.39 on test, 0.26 on validation) — frequently confused with other classes, consistent with monocytes and lymphocytes/neutrophils sharing overlapping morphological traits.
- **Eosinophil** has high recall but low precision — the model over-predicts this class, pulling in false positives from other types.
- Test accuracy (0.66) exceeds validation accuracy (0.55), most likely because the test evaluation cell reuses the *entire* `Train`-fitted model/scaler while testing against a differently-sized/composed set — the two numbers come from different runs/splits and aren't directly comparable in a rigorous sense.

---

## How to Run

### Prerequisites
- Python 3.x
- Jupyter Notebook / JupyterLab (or VS Code with the Jupyter extension)

### Setup
```bash
pip install opencv-python numpy matplotlib scikit-learn seaborn joblib
```

### Steps
1. Open [`main.ipynb`](main.ipynb) in Jupyter/VS Code.
2. **Update the hardcoded dataset paths** in Cell 3, Cell 26, and Cell 32 to point at your local `wbc_data/Train`, `wbc_data/Test` (and optionally `wbc_data/Train_small`) folders — the notebook currently hardcodes an author-specific absolute path.
3. Run all cells top to bottom:
   - Cells 1–22 define the preprocessing/feature-extraction pipeline and can be run without any dataset (Cell 14 just demonstrates nucleus extraction on 5 sample images).
   - Cell 26 builds the training dataset (this iterates over all 500 training images and is the most time-consuming step, since `CCA_8`, `local_binary_pattern`, and `extract_hog` are all pure-Python/NumPy pixel-loop implementations — expect this to take a while per image), trains the SVM, and prints validation metrics.
   - Cell 28 saves the trained model/scaler/label map to disk.
   - Cells 30–32 reload the saved artifacts and evaluate on the `Test` folder.
4. To classify a **new single image** without retraining, load the saved artifacts (Cell 30) and call:
   ```python
   img = cv2.imread("path/to/image.jpg", 0)
   feats = extract_all_features(img)
   feats_scaled = scaler.transform([feats])
   pred = model.predict(feats_scaled)
   print(inv_label_map[pred[0]])
   ```

---

## Dependencies

| Library        | Purpose                                                    |
|-----------------|-------------------------------------------------------------|
| `opencv-python` | Image I/O, thresholding, morphology, contours, Sobel, resize |
| `numpy`         | Array math, FFT (Butterworth filtering), histograms          |
| `matplotlib`    | Visualization (original vs. nucleus image, plots)            |
| `scikit-learn`  | `SVC`, `StandardScaler`, `train_test_split`, metrics          |
| `seaborn`       | Confusion matrix heatmap visualization                        |
| `joblib`        | Model/scaler serialization                                    |
| `json`          | Label map serialization                                       |
| `copy` / `os`   | Standard library — deep copies, filesystem traversal          |

---

## Known Issues & Limitations

- **Hardcoded absolute paths:** Dataset paths in Cells 3, 26, and 32 are hardcoded to a specific machine and must be edited before the notebook will run elsewhere.
- **No model artifacts committed:** `wbc_classifier_model.pkl`, `wbc_scaler.pkl`, and `wbc_label_map.json` are not present in the repository — they are generated only after running the training cells locally.
- **Performance:** All core algorithms (connected-component labeling, LBP, HOG) are implemented with raw Python `for` loops over pixels/cells rather than vectorized NumPy or OpenCV built-ins, making feature extraction slow, especially over the full 500-image training set.
- **Nucleus-centering assumption:** `CCA_8`'s nucleus-selection heuristic assumes the nucleus is at or near the image center (`a[x/2, y/2]`) with a bottom-right fallback scan; this can misidentify the nucleus on images where the cell/nucleus is off-center or where multiple nuclei-like blobs exist near the center.
- **Moderate/uneven accuracy:** Overall classifier accuracy is modest (55–66%), with **Monocyte** and **Eosinophil** classes performing considerably worse than **Basophil** and **Lymphocyte**, indicating the current hand-crafted feature set doesn't fully separate all five classes.
- **Very high-dimensional LBP feature:** Using the full flattened 128×128 LBP map (16,384 dims) rather than an LBP *histogram* is unusual — it makes the feature vector extremely high-dimensional relative to the training set size (500 samples), risking overfitting/curse-of-dimensionality, and is sensitive to any spatial misalignment between images.
- **`Train_small` folder is unused:** The notebook only references `Train` and `Test`; the `Train_small` (250-image) folder exists in the dataset but isn't wired into any cell.
- **Mislabeled section header:** The "Hough Transformation" markdown heading (Cell 17) precedes the HOG feature extractor, not an actual Hough transform implementation.

## Possible Future Improvements

- Vectorize LBP/HOG/CCA using NumPy strides or OpenCV/`skimage` built-ins to drastically speed up feature extraction.
- Replace the flattened LBP map with a proper LBP **histogram** (e.g., uniform patterns, 59-bin histogram) to reduce dimensionality and improve generalization.
- Add cross-validation and hyperparameter search (`GridSearchCV`) for the SVM's `C`/`gamma`.
- Parameterize dataset paths (e.g., via relative paths or a config/CLI argument) instead of hardcoding.
- Wire in the unused `Train_small` split, e.g. as a quick-iteration training set.
- Consider augmenting hand-crafted features with a lightweight CNN embedding for comparison against the classical pipeline.
- Add per-image visualization/export of the extracted nucleus mask for qualitative pipeline debugging.
