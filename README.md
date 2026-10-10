# CLAHE-Enhanced Thermal Human Candidate Detection for Drone-Based Search and Rescue

A Python-based image processing project that explores the use of **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to enhance thermal images and detect potential human regions for search-and-rescue applications.

## Overview

Thermal imaging can help locate people in low-visibility environments. However, low contrast, background heat sources, and image noise can make potential human regions difficult to distinguish.

This project implements a classical computer vision pipeline using Python and OpenCV. It compares basic threshold-based detection with CLAHE-enhanced image processing to investigate how local contrast enhancement affects candidate detection performance.

The system processes individual thermal images and evaluates detection performance using precision, recall, F1-score, and runtime.

**Note:** This is a classical image-processing prototype, not a trained deep-learning model. It identifies candidate regions based on image intensity and contour properties; it does not reliably confirm that a detected region contains a human.

## Objectives

* Enhance local contrast in thermal images using CLAHE.
* Implement image preprocessing and noise reduction.
* Detect potential human regions using thresholding and contour analysis.
* Compare baseline thresholding against CLAHE-enhanced processing.
* Evaluate performance using precision, recall, F1-score, and execution time.
* Investigate the trade-off between detecting more potential subjects and generating false positives.

## Technologies Used

* **Python** — programming language
* **OpenCV** — image processing and computer vision
* **NumPy** — numerical operations
* **Matplotlib** — visualization of images and results

## Methodology

The general image-processing pipeline consists of the following stages:

1. **Image Loading:** Load a thermal image from the dataset.
2. **Grayscale Conversion:** Convert the image to grayscale when necessary.
3. **CLAHE Enhancement:** Improve local contrast while limiting excessive amplification of noise.
4. **Noise Reduction:** Apply filtering where configured.
5. **Thresholding:** Convert the image into a binary mask based on pixel intensity.
6. **Morphological Operations:** Use opening and closing to reduce small artifacts and refine candidate regions.
7. **Contour Detection:** Extract connected regions from the binary image.
8. **Candidate Filtering:** Filter regions using properties such as area, width, height, and aspect ratio.
9. **Performance Evaluation:** Compare predicted bounding boxes with ground-truth annotations using Intersection over Union (IoU).

## Dataset

The project uses thermal images with YOLO-format bounding-box annotations.

The dataset directory is organized as follows:

```text
dataset/
└── images/
    └── thermal dataset/
        ├── train/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
```

Each annotation file follows the YOLO format:

```text
class_id x_center y_center width height
```

The bounding-box coordinates are normalized relative to the image dimensions. In the dataset used during development, class `0` represents a human.

Dataset statistics checked during development:

* **Training images:** 12,023
* **Testing images:** 3,463
* **Training annotation files:** 12,023
* **Testing annotation files:** 3,463

The dataset files are not included in this repository by default. Users should obtain the dataset from its original source and follow its licensing and usage requirements.

## Project Structure

```text
SIP/
├── dataset/
│   └── images/
│       └── thermal dataset/
│           ├── train/
│           │   ├── images/
│           │   └── labels/
│           └── test/
│               ├── images/
│               └── labels/
├── src/
│   ├── clahe.py
│   ├── preprocessing.py
│   ├── stage2.py
│   ├── detection.py
│   ├── stage3.py
│   ├── comparison.py
│   ├── check_dataset.py
│   ├── evaluate.py
│   ├── evaluate_clahe.py
│   └── tune_threshold.py
├── test.jpg
├── requirements.txt
└── README.md
```

*This structure reflects the development project. Adjust it to match the files included in your repository. The dataset, virtual environment, and generated outputs may be excluded from Git.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Replace the placeholders with your actual GitHub username and repository name.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file containing:

```text
opencv-python
numpy
matplotlib
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your scripts use additional libraries, include them in `requirements.txt`.

## Running the Project

Execute the commands from the repository root directory.

### Run CLAHE enhancement

```bash
python src/clahe.py
```

### Run preprocessing

```bash
python src/preprocessing.py
```

### Run candidate detection

```bash
python src/detection.py
```

### Compare baseline and CLAHE

```bash
python src/comparison.py
```

### Check dataset files and annotations

```bash
python src/check_dataset.py
```

### Evaluate detection performance

```bash
python src/evaluate.py
python src/evaluate_clahe.py
```

### Tune detection parameters

```bash
python src/tune_threshold.py
```

These commands assume the script names and entry points match the project structure above. Check the relevant Python file if a command requires different paths or parameters.

## Evaluation Metrics

The detector can be evaluated by comparing predicted bounding boxes with annotated ground-truth boxes. With an IoU threshold of 0.5:

* **True Positive (TP):** A predicted bounding box successfully matches a ground-truth box.
* **False Positive (FP):** A predicted box does not match a ground-truth box.
* **False Negative (FN):** A ground-truth object is not detected.

The evaluation metrics are:

**Precision**

Precision measures the proportion of predicted detections that are correct.

`Precision = TP / (TP + FP)`

**Recall**

Recall measures the proportion of annotated objects successfully detected.

`Recall = TP / (TP + FN)`

**F1-score**

The F1-score is the harmonic mean of precision and recall.

`F1 = 2 × Precision × Recall / (Precision + Recall)`

**Runtime**

Runtime measures the time taken to process the evaluation dataset. Comparisons should use the same hardware, dataset split, IoU threshold, and matching procedure.

## Results

The following results were measured during earlier development experiments.

| Configuration                                                       | Precision | Recall | F1-score |
| ------------------------------------------------------------------- | --------: | -----: | -------: |
| Baseline thresholding: threshold 140, minimum area 600              |     5.43% | 11.68% |    7.41% |
| CLAHE: clip limit 2.0, tile size 8, threshold 140, minimum area 800 |     3.58% | 14.96% |    5.78% |

The baseline runtime was approximately **51.98 seconds**, while the measured CLAHE configuration took approximately **55.5 seconds**.

### Discussion

The measured CLAHE configuration achieved higher recall, meaning that it detected more of the annotated objects. However, it also produced more false positives, resulting in lower precision and a lower F1-score than the baseline.

These findings indicate that contrast enhancement alone does not guarantee better overall detection performance. Parameter tuning and improved candidate-region filtering may be necessary to reduce false detections.

**Experimental status:** These values come from earlier measured configurations, not from a completed 360-configuration CLAHE experiment. Any projected or estimated values must be clearly identified as hypothetical and must not be presented as measured results.

## Limitations

* Contour-based detection identifies candidate regions rather than reliably classifying humans.
* Background heat sources and image noise can produce false positives.
* Performance depends on image quality, threshold values, CLAHE parameters, and contour filters.
* Parameter tuning on the test set can lead to overfitting; a separate validation set is preferable.
* Runtime depends on hardware, image dimensions, and implementation details.
* The current implementation processes individual images rather than videos and does not perform temporal tracking.
* The system has not been established as suitable for real-world rescue operations.

## Future Improvements

* Tune parameters using a separate validation dataset.
* Improve candidate filtering to reduce false positives.
* Compare CLAHE with other thermal image enhancement techniques.
* Generate visual comparisons of original, enhanced, and detected images.
* Evaluate additional metrics and test configurations.
* Explore a trained object-detection model as a separate future extension.

## Academic and Educational Use

This project demonstrates the application of digital image processing and classical computer vision to a search-and-rescue problem. It is intended for educational and research purposes.

## Disclaimer

This project is an experimental prototype and is not a validated search-and-rescue system. Its outputs should not be used as the sole basis for operational or emergency decisions.
