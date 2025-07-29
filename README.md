<h1 align="center">BONBID-Ensemble</h1>

<p align="center">
  <img src="https://img.shields.io/badge/BONBIDE--HIE-MICCAI_2023_3rd_Place-blue" alt="MICCAI 2023 Badge"/>
</p>


<p align="center"><strong>Ensemble Framework for Neonatal Brain Injury Prediction</strong><br>
<i>🥉 Ranked 3rd in the <a href="https://bonbid-hie2023.grand-challenge.org">MICCAI 2023 BONBID‑HIE Challenge</a></i></p>

---

## 🔬 Challenge Background

BONBID‑HIE (Boston Neonatal Brain Injury Dataset for Hypoxic‑Ischemic Encephalopathy) is the official MICCAI 2023 challenge focused on lesion segmentation in neonatal MRI. The challenge dataset, evaluation protocol, and leaderboards are hosted on the [Grand Challenge portal](https://bonbid-hie2023.grand-challenge.org).

Our paper, **"Enhancing Lesion Segmentation in the BONBID‑HIE Challenge: An Ensemble Strategy"**, was accepted into the official MICCAI 2023 proceedings. It introduces a transformer-based ensemble approach—leveraging architectures like Swin-UNETR—that achieved top-tier performance in the challenge leaderboard.

---

## 🗂️ Repository Preview

```
bonbid‑ensemble/
├── train.py               # Ensemble training
├── train_unetr.py         # UNETR-specific trainer
├── test.py                # Inference pipeline
├── evaluation.py          # Computes Dice, Hausdorff, etc.
├── convert_to_nii.py      # Converts raw images to NIfTI format
├── transforms.py          # Augmentations + preprocessing
├── model.py               # Network architecture definitions
├── Dockerfile             # Reproducible containerized build
├── requirements.in        # Python dependencies
└── README.md              
```

---

## 🚀 Quick Setup

### 🐍 Local Installation
```bash
git clone https://github.com/es15326/bonbid-ensemble.git
cd bonbid-ensemble
pip install -r requirements.in
```

### 🐳 Docker (Recommended for Reproducibility)
```bash
docker build -t bonbid-ensemble .
```

---

## 📦 Preparing the Challenge Dataset

The dataset must comply with the BONBID‑HIE format. Use:

```bash
python convert_to_nii.py --input path/to/raw_data --output path/to/nifti_data
```

Ensure patient folders and filenames align with the official structure.

---

## 🏋️‍♀️ Model Training

```bash
python train.py --config configs/train_config.yaml
# or specific UNETR training:
python train_unetr.py --config configs/unetr_config.yaml
```

---

## 🔍 Inference & Evaluation

```bash
python test.py --model_path checkpoints/best_model.pth --output results/
python evaluation.py --pred_dir results/ --gt_dir ground_truth/
```

Use helper scripts:

```bash
bash test.sh
bash export.sh
```

---

## 📊 Challenge Metrics Summary

| Metric         | Value               | Rank   |
|----------------|---------------------|--------|
| **Mean Dice**  | 0.5741 ± 0.2387      | 🥉 3rd |
| **MASD**       | 2.6668 ± 3.4076     | —      |
| **NSD**        | 0.7338 ± 0.2491     | —      |
| **Hausdorff**  | 4.3 mm              | —      |

> These scores are based on official BONBID‑HIE leaderboard evaluation at MICCAI 2023. Our ensemble achieved the 3rd highest Mean Dice, a critical segmentation metric for clinical tasks.

## 📚 Citation

Please cite our work and the official MICCAI paper if you use this repository:

```bibtex
@inproceedings{soltanikazemi2025ensemble,
  title={Enhancing Lesion Segmentation in the BONBID‑HIE Challenge: An Ensemble Strategy},
  author={Soltani Kazemi, Imad Eddine and Toubal, Elham and Rahmon, Gani and others},
  booktitle={AI for Brain Lesion Detection and Trauma Video Action Recognition – 1st BONBID‑HIE Lesion Segmentation Challenge at MICCAI 2023},
  series={Lecture Notes in Computer Science}, volume={14567},
  pages={14–22},
  year={2025},
  publisher={Springer}
}
```

---

## 🤝 Acknowledgements

- [BONBIDE‑HIE Challenge 2023](https://bonbid-hie2023.grand-challenge.org)
- The neonatal imaging AI research community
- Projects like PyTorch, MONAI, and related imaging toolkits

---

<p align="center"><em>Empowering neonatal care with robust AI research.</em></p>
