# ML-Based Magnetic Map Matching

A machine learning reimplementation of the **Mag-Match** magnetic localization pipeline. The project replaces the computationally expensive Gaussian Process (GP) inference and Local Reference Frame (LRF) estimation stages with lightweight neural networks (**MagFieldNet** and **LRFNet**), enabling significantly faster map matching while maintaining comparable accuracy.

The project includes both:

* The original analytical (GP-based) Mag-Match implementation.
* The proposed machine learning based version.
* Scripts for running experiments on the KI Building dataset.

## Directory Structure

```text
.
├── mag_match.py          # Original GP-based Mag-Match implementation
├── mag_match_ml.py       # ML-based Mag-Match (MagFieldNet + LRFNet)
└── run_ki_building.py    # Run experiments on the KI Building dataset
```

## Running the Classical Version

```bash
python mag_match.py                       # one trial, gravity-aligned
python mag_match.py --non-gravity-aligned # one trial, x-axis rotation
python mag_match.py --trials 5            # 5-trial Monte-Carlo, RMSE report
python mag_match.py --plot                # also writes mag_match_demo.png
```

## Running the ML Version

```bash
python mag_match_ml.py              # train + evaluate
python mag_match_ml.py --skip-train # use cached weights
python mag_match_ml.py --epochs 200 # shorter training run
python mag_match_ml.py --device cpu # force CPU
```

## KI Building Dataset

To run the ML pipeline on the KI Building dataset:

```bash
python run_ki_building.py
```

## Overview

The pipeline extracts magnetic field features from 3-axis magnetometer measurements, detects keypoints, computes descriptors, matches maps, and estimates the relative SE(3) transformation between two traversals of the same environment.

Compared to the analytical implementation, the ML-based approach achieves similar localization accuracy while providing a significant speedup in feature extraction and map matching.