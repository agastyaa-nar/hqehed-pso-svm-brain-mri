# GitHub Project Plan — `hqehed-pso-svm-brain-mri`

## Repository Name

```
hqehed-pso-svm-brain-mri
```

**Deskripsi:** OOP Python implementation of HQEHED-AMT quantum edge detection
+ PSO-SVM segmentation for brain MRI tumor analysis (Rengasamy et al., 2025).

**Topics/Tags:** `quantum-computing` · `qiskit` · `medical-imaging` · `brain-mri` ·
`svm` · `pso` · `edge-detection` · `tumor-segmentation` · `python`

---

## Branch Strategy

```
main          ← stable releases
develop       ← integration branch
feature/*     ← per-feature branches
```

---

## Commit + Push Plan (per Fitur)

---

### 🔧 FASE 0 — Inisialisasi

```bash
git init
git remote add origin https://github.com/<user>/hqehed-pso-svm-brain-mri.git
git checkout -b main
```

#### Commit 0.1 — Project scaffolding
```bash
git add .gitignore README.md requirements.txt
git commit -m "chore: initial project scaffold

- Add .gitignore for Python/results/dataset
- Add README with project overview and quick start
- Add requirements.txt with all dependencies"
git push -u origin main
```

---

### 🏗️ FASE 1 — Core Infrastructure

```bash
git checkout -b feature/core-config-dataset
```

#### Commit 1.1 — Config dataclass
```bash
git add src/core/config.py src/core/__init__.py src/__init__.py
git commit -m "feat(core): add Config dataclass with all hyperparameters

- Centralize all parameters (PSO, SVM, HQEHED, dataset paths)
- Support from_dict() and to_dict() for YAML/JSON serialization
- Auto-create output_dir on init"
```

#### Commit 1.2 — Dataset loader & preprocessor
```bash
git add src/core/dataset.py
git commit -m "feat(core): add DatasetLoader and Preprocessor

- DatasetLoader.load_pairs(): scan *_mask.* pairs recursively
- DatasetLoader.load_pairs_by_class(): group by tumor type
- DatasetLoader.load_image_mask(): grayscale + resize + normalize
- Preprocessor.process(): Gaussian blur (Section 3.2)"
```

#### Commit 1.3 — Classic edge detectors
```bash
git add src/core/classic_edge.py
git commit -m "feat(core): add classic edge detectors (Canny, Sobel, Prewitt, LoG)

- Unified EdgeDetector base class with detect() interface
- EdgeDetectorFactory registry for managing all detectors
- Used as comparison baselines in evaluation"
```

```bash
git push origin feature/core-config-dataset
# PR: feature/core-config-dataset → develop
```

---

### ⚛️ FASE 2 — Quantum Edge Detection (HQEHED-AMT)

```bash
git checkout -b feature/quantum-edge-detection
```

#### Commit 2.1 — QPIE encoder
```bash
git add src/core/quantum_edge.py
git commit -m "feat(quantum): implement QPIEEncoder via Qiskit Statevector (Eq. 2.4)

- Encode pixel row as L2-normalized quantum amplitudes
- Pad to next power-of-2 for qubit allocation
- Returns Re(sv)[:N] — QPIE amplitude vector"
```

#### Commit 2.2 — Hadamard edge amplitude
```bash
git commit -m "feat(quantum): add HadamardEdgeAmplitude (Eq. 2.8-2.16)

- Forward cyclic difference on QPIE state via Qiskit
- Computes diff[i] = sv[i] - sv[(i+1) % N_pad]
- Removes false edges at padding boundary
- Mathematically equivalent to H-ancilla + Ctrl-UD circuit"
```

#### Commit 2.3 — AMT thresholding
```bash
git commit -m "feat(quantum): add AdaptiveMeanThreshold (Eq. 2.17-2.20)

- Dual threshold: T = gamma * mean(|diff|), T_ = -gamma_neg * mean(|diff|)
- Binary edge map output {0, 1}"
```

#### Commit 2.4 — PPQEC error correction
```bash
git commit -m "feat(quantum): add PPQECCorrector post-quantum error correction (Section 2.8.3)

- Correct FP/FN across sequential crops
- Decision based on quantum probability P = |diff|^2"
```

#### Commit 2.5 — Full HQEHED-AMT pipeline
```bash
git commit -m "feat(quantum): integrate HQEHEDPipeline — full HQEHED-AMT (Section 3.3)

- Horizontal + vertical scan with crop-wise processing
- OR fusion of both scan directions
- PPQEC applied per scan direction
- Accepts crop_size, gamma, gamma_neg overrides"
```

```bash
git push origin feature/quantum-edge-detection
# PR: feature/quantum-edge-detection → develop
```

---

### 🔍 FASE 3 — Feature Extraction

```bash
git checkout -b feature/feature-extraction
```

#### Commit 3.1 — 27-D feature extractor
```bash
git add src/models/features.py src/models/__init__.py
git commit -m "feat(models): add FullFeatureExtractor — 27-D feature vectors (Section 3.4)

- 9 feature maps: intensity, mean, std, grad_mag, grad_cos, grad_sin,
  edge_flag, edge_density, dist_to_edge
- Each map: [center, local_mean(7x7), local_std(7x7)] = 3 stats
- sample_features(): 1:3 tumor:non-tumor sampling
- predict_full_image(): vectorized pixel-wise inference"
```

#### Commit 3.2 — 18-D ablation variant
```bash
git commit -m "feat(models): add NoEdgeFeatureExtractor — 18-D ablation variant

- Excludes edge_flag, edge_density, dist_to_edge feature maps
- Used to quantify contribution of HQEHED edge features
- Inherits FullFeatureExtractor interface"
```

```bash
git push origin feature/feature-extraction
# PR: feature/feature-extraction → develop
```

---

### 🧬 FASE 4 — PSO Optimizer & SVM Segmenter

```bash
git checkout -b feature/pso-svm-model
```

#### Commit 4.1 — PSO optimizer
```bash
git add src/models/pso_optimizer.py
git commit -m "feat(models): implement PSOOptimizer for SVM-RBF hyperparameter search

- Search in log10(C) and log10(gamma) space (Eq. 2.27-2.28)
- Fitness function: Dice coefficient on validation set
- Configurable particles, iterations, w, c1, c2 via Config
- Returns (C_opt, g_opt, convergence_history)"
```

#### Commit 4.2 — SVM segmenter + morphological post-processing
```bash
git add src/models/segmentation.py
git commit -m "feat(models): add SVMSegmenter with morphological post-processing

- MorphologicalPostProcessor: MORPH_OPEN + connected-component filter
  (cc_min_size, cc_max_blobs from Config)
- SVMSegmenter.fit(): sample pixels, split 70/30, train SVC
- SVMSegmenter.predict_full(): pixel-wise inference on full image
- Uses globally optimized C_opt, g_opt from PSO phase"
```

```bash
git push origin feature/pso-svm-model
# PR: feature/pso-svm-model → develop
```

---

### 📊 FASE 5 — Evaluation & Grid Search

```bash
git checkout -b feature/evaluation
```

#### Commit 5.1 — Core metrics
```bash
git add src/evaluation/metrics.py src/evaluation/__init__.py
git commit -m "feat(evaluation): add EdgeMetrics, SegmentationMetrics, ImageQualityMetrics

- EdgeMetrics.figure_of_merit(): Pratt's FOM (Eq. 2.39)
- EdgeMetrics.compute(): Precision, Recall, F1, TP/FP/FN
- SegmentationMetrics.compute(): Accuracy, Sensitivity, Specificity, Dice, IoU
- ImageQualityMetrics.mse_psnr(): MSE and PSNR with R=255 (Eq. 2.18-2.19)"
```

#### Commit 5.2 — Grid search tuner
```bash
git add src/evaluation/grid_search.py
git commit -m "feat(evaluation): add GridSearchTuner for gamma/crop_size selection

- Exhaustive grid search over user-defined parameter ranges
- Scoring criterion: mean PSNR (higher is better)
- print_top(): formatted top-N results table
- Returns sorted list of dicts for downstream use"
```

#### Commit 5.3 — Comparative evaluator
```bash
git add src/evaluation/comparative.py
git commit -m "feat(evaluation): add ComparativeEvaluator — per-class multi-method analysis

- Evaluates glioma / meningioma / pituitary separately
- Per-image SVM retrain with PSO-optimized C*, g*
- Computes FOM + segmentation metrics per (class, method)
- aggregate(): averages results across images
- Supports any callable edge detector via detectors dict"
```

```bash
git push origin feature/evaluation
# PR: feature/evaluation → develop
```

---

### 🎨 FASE 6 — Visualization

```bash
git checkout -b feature/visualization
```

#### Commit 6.1 — Panel, error map, PSO convergence, metrics bar
```bash
git add src/visualization/visualizer.py src/visualization/__init__.py
git commit -m "feat(viz): add all dark-theme visualization components

- PanelVisualizer: 5-column MRI result panel
- ErrorMapVisualizer: TP/FP/FN overlay + error RGB map
- PSOConvergenceVisualizer: convergence curve with exploration shading
- MetricsBarVisualizer: grouped bar for Dice/IoU/FOM etc.
- Consistent dark GitHub-style theme (#0d1117 background)"
```

```bash
git push origin feature/visualization
# PR: feature/visualization → develop
```

---

### 🧪 FASE 7 — Tests

```bash
git checkout -b feature/tests
```

#### Commit 7.1 — Metrics unit tests
```bash
git add tests/test_metrics.py tests/__init__.py
git commit -m "test: add unit tests for EdgeMetrics, SegmentationMetrics, ImageQualityMetrics

- FOM perfect/empty/no-detection cases
- Precision/Recall formula verification
- Dice and IoU edge cases
- MSE=0 and PSNR formula check"
```

#### Commit 7.2 — Dataset & config tests
```bash
git add tests/test_dataset.py
git commit -m "test: add unit tests for Config and Preprocessor

- Config default values and from_dict/to_dict roundtrip
- Preprocessor output shape and dtype
- Gaussian blur variance reduction check"
```

#### Commit 7.3 — Feature extractor tests
```bash
git add tests/test_features.py
git commit -m "test: add unit tests for FullFeatureExtractor and NoEdgeFeatureExtractor

- Feature vector length (27-D full, 18-D no-edge)
- Feature map key presence
- Finite value guarantee"
```

```bash
git push origin feature/tests
# PR: feature/tests → develop
```

---

### 🚀 FASE 8 — CLI Pipeline Runner

```bash
git checkout -b feature/pipeline-runner
```

#### Commit 8.1 — End-to-end runner
```bash
git add scripts/run_pipeline.py
git commit -m "feat(scripts): add run_pipeline.py — CLI end-to-end runner

- argparse: --data_root, --output_dir, --max_images, --gamma,
  --crop_size, --skip_grid, --skip_pso
- Phases: load → grid search → quick-viz → PSO → per-image SVM
- Auto-saves all visualizations to output_dir"
```

```bash
git push origin feature/pipeline-runner
# PR: feature/pipeline-runner → develop
```

---

### 🔀 FASE 9 — Merge ke Main & Release

```bash
git checkout main
git merge develop --no-ff -m "release: v1.0.0 — full HQEHED-AMT + PSO-SVM OOP pipeline"
git tag -a v1.0.0 -m "v1.0.0: Initial release — quantum edge detection + PSO-SVM segmentation"
git push origin main --tags
```

---

## Ringkasan Commit Timeline

| Fase | Branch | Commits | Fitur |
|------|--------|---------|-------|
| 0 | main | 1 | Scaffold, README, requirements |
| 1 | feature/core-config-dataset | 3 | Config, DatasetLoader, classic edges |
| 2 | feature/quantum-edge-detection | 5 | QPIE, H+UD, AMT, PPQEC, pipeline |
| 3 | feature/feature-extraction | 2 | 27-D extractor, 18-D ablation |
| 4 | feature/pso-svm-model | 2 | PSO optimizer, SVMSegmenter |
| 5 | feature/evaluation | 3 | Metrics, grid search, comparative |
| 6 | feature/visualization | 1 | All visualizers |
| 7 | feature/tests | 3 | Unit tests (metrics, dataset, features) |
| 8 | feature/pipeline-runner | 1 | CLI runner |
| 9 | main | 1 | Merge + tag v1.0.0 |
| **Total** | | **22** | |
