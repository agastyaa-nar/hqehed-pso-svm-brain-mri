# HQEHED-AMT + PSO-SVM Brain Tumor MRI Segmentation

> OOP Python implementation of *Rengasamy et al. (2025)*:
> **"Hybrid Quantum Edge-Hadamard Encoder Detector with Adaptive Mean Thresholding
> and PSO-Optimized SVM for Brain MRI Tumor Segmentation"**

---

## Project Structure

```
hqehed_pso_svm/
├── src/
│   ├── core/
│   │   ├── config.py           # Centralized Config dataclass
│   │   ├── dataset.py          # DatasetLoader + Preprocessor
│   │   ├── quantum_edge.py     # HQEHED-AMT pipeline (Qiskit)
│   │   └── classic_edge.py     # Canny / Sobel / Prewitt / LoG
│   ├── models/
│   │   ├── features.py         # 27-D & 18-D feature extractors
│   │   ├── pso_optimizer.py    # PSO for SVM hyperparameter tuning
│   │   └── segmentation.py     # SVMSegmenter + morphological post-proc
│   ├── evaluation/
│   │   ├── metrics.py          # FOM, Dice, IoU, Precision/Recall
│   │   ├── grid_search.py      # GridSearchTuner
│   │   └── comparative.py      # ComparativeEvaluator (per tumor class)
│   └── visualization/
│       └── visualizer.py       # All dark-theme matplotlib plots
├── scripts/
│   └── run_pipeline.py         # CLI end-to-end runner
├── tests/
│   ├── test_metrics.py
│   ├── test_dataset.py
│   └── test_features.py
├── configs/                    # YAML config files (optional)
├── results/                    # Output images and logs
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/<your-username>/hqehed-pso-svm-brain-mri.git
cd hqehed-pso-svm-brain-mri
pip install -r requirements.txt

# 2. Prepare dataset (Kaggle: nikhilroxtomar/brain-tumor-segmentation)
# Expected structure:
#   dataset/<tumor_type>/images/<file>.png
#   dataset/<tumor_type>/masks/<file>.png

# 3. Run full pipeline
python scripts/run_pipeline.py --data_root ../dataset --output_dir ./results

# 4. Skip grid search for speed
python scripts/run_pipeline.py --skip_grid --gamma 0.3 --crop_size 64

# 5. Run tests
pytest tests/ -v
```

---

## Pipeline Overview

```
Dataset
  └── DatasetLoader + Preprocessor (Gaussian blur)
        └── HQEHEDPipeline (QPIE → H+UD → AMT → OR → PPQEC)
              └── GridSearchTuner  (select best γ, crop_size)
                    └── PSOOptimizer (search C*, γ* for SVM-RBF)
                          └── SVMSegmenter (per-image train + predict)
                                └── MorphologicalPostProcessor
                                      └── ComparativeEvaluator
                                            └── Visualizer
```

---

## Key Classes

| Class | Module | Role |
|-------|--------|------|
| `Config` | `core.config` | All hyperparameters as a dataclass |
| `DatasetLoader` | `core.dataset` | Scan and load image-mask pairs |
| `HQEHEDPipeline` | `core.quantum_edge` | Full quantum edge detection |
| `QPIEEncoder` | `core.quantum_edge` | QPIE encoding via Qiskit |
| `HadamardEdgeAmplitude` | `core.quantum_edge` | H+UD gradient (Eq. 2.8–2.16) |
| `AdaptiveMeanThreshold` | `core.quantum_edge` | AMT (Eq. 2.17–2.20) |
| `PPQECCorrector` | `core.quantum_edge` | Post-quantum error correction |
| `FullFeatureExtractor` | `models.features` | 27-D feature vectors |
| `NoEdgeFeatureExtractor` | `models.features` | 18-D ablation variant |
| `PSOOptimizer` | `models.pso_optimizer` | PSO (C, γ) search |
| `SVMSegmenter` | `models.segmentation` | RBF-SVM + morphological filter |
| `EdgeMetrics` | `evaluation.metrics` | FOM, Precision, Recall, F1 |
| `SegmentationMetrics` | `evaluation.metrics` | Dice, IoU, Acc, Sen, Spe |
| `GridSearchTuner` | `evaluation.grid_search` | γ / crop_size selection |
| `ComparativeEvaluator` | `evaluation.comparative` | Per-class comparison |

---

## Reference

Rengasamy, D., et al. (2025). *Novel Error-Corrected Quantum Unitary Gate Hadamard
Edge Detection for Brain MRI Segmentation*. [Paper details omitted for brevity]