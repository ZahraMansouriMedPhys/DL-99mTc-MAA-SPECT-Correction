# DL-based Attenuation and Scatter Correction for 99mTc-MAA SPECT Imaging

This repository provides deep learning models developed for **CT-free attenuation correction (AC)**, **Monte Carlo-based scatter correction (SC)**, and **combined attenuation and scatter correction (ASC)** in **99mTc-MAA SPECT imaging**, using a modified **Swin UNETR** architecture.

These models were trained to improve **quantitative SPECT imaging**, **pre-therapy dosimetry**, and **treatment planning** for patients undergoing **90Y Selective Internal Radiation Therapy (SIRT)**.

🧠 **Read the reference publication for more information:**

Mansouri Z., Salimi Y., Bianchetto Wolf N., Hajianfar G., Mainta I., Garibotto V., Zaidi H.

*Deep learning-guided attenuation and scatter correction of 99mTc-MAA SPECT images: towards quantitative analysis in 90Y-SIRT*

*Annals of Nuclear Medicine*, 2026

DOI: https://doi.org/10.1007/s12149-025-02152-2

---

## 🧪 Background

**99mTc-MAA SPECT imaging** is routinely performed before radioembolization to estimate dose distribution and support personalized treatment planning. However, quantitative accuracy is affected by photon attenuation and scatter.

Conventional attenuation correction relies on CT-derived attenuation maps, whereas accurate scatter correction often requires computationally demanding Monte Carlo simulations.

This work introduces fast and robust deep learning-based alternatives using a Swin UNETR architecture trained on a large clinical dataset (**n = 222 patients**) to perform:

* CT-free attenuation correction (AC)
* Monte Carlo-based scatter correction (SC)
* Joint attenuation and scatter correction (ASC)

---

## 🏥 Clinical Context

These models may be particularly beneficial in:

* Standalone SPECT systems without CT
* Centers lacking GPU resources for Monte Carlo simulations
* Reducing processing times
* Improving quantitative SPECT imaging
* Personalized treatment planning and dosimetry for SIRT

---

## 📥 Download Trained Models

The three trained model weights are stored in [`models/`](models/) using Git LFS.
Install [Git LFS](https://git-lfs.com/), then download the repository and weights:

```bash
git lfs install
git clone https://github.com/ZahraMansouriMedPhys/DL-99mTc-MAA-SPECT-Correction.git
cd DL-99mTc-MAA-SPECT-Correction
git lfs pull
```

Each checkpoint is 70,412,185 bytes. SHA-256 checksums are recorded in
[`models/SHA256SUMS`](models/SHA256SUMS). A small text file beginning with
`version https://git-lfs.github.com/spec/v1` is an LFS pointer; run `git lfs pull`
to download the actual weights.

The original Google Drive location is:

https://drive.google.com/drive/folders/1kBq7gA8YaatLOVIYUyv6eIHMwV7BhWGx?usp=drive_link

Access to the Google Drive copy is managed by the authors.

Available models:

* **AC Model** — CT-free attenuation correction
* **SC Model** — Monte Carlo-based scatter correction
* **ASC Model** — Joint attenuation and scatter correction

---

## 🚀 Installation

```bash
pip install git+https://github.com/ZahraMansouriMedPhys/DL-99mTc-MAA-SPECT-Correction.git
```

This installs the Python modules and dependencies. Download the checkpoints
separately using the Git LFS commands above; they are not included in the Python
package.

### Inference

Use Python 3.10 or newer. The same `DL_inference.py` interface as the 90Y
repository now loads the three MAA weight files:

```python
from DL_inference import model_inference_regression

result = model_inference_regression(
    model_url="models/AC_Model.pth",  # or SC_Model.pth / ASC_Model.pth
    list_images=["input.nii.gz"],
    predict_directory="predictions",
    device="cpu",  # or "auto" / "cuda"
    suffix="_AC",
)
```

AC and ASC take non-corrected SPECT; SC takes attenuation-corrected SPECT.
Outputs retain the input image shape and spatial coordinates. Existing files
are not overwritten.

The weights do not contain training preprocessing. Defaults keep native spacing,
orientation, and intensities, with raw network output. Optional `pixdim`,
`orientation`, `input_range=(min,max)` (maps to `[0,1]`), and
`output_range=(min,max)` (maps output `[0,1]` back to that range) let you specify
the preparation explicitly. `normalize_features=True`, `downsample="merging"`,
and 64-voxel sliding windows are runtime assumptions, not recovered training
settings. All three models run on synthetic images; quantitative accuracy
cannot be verified without the original preprocessing and reference images.

---

## 📊 Dataset

* 222 patients undergoing 99mTc-MAA SPECT imaging prior to 90Y-SIRT
* Five-fold cross-validation training framework
* Independent external test set
* Quantitative evaluation through voxel-wise and organ-level dosimetry

---

## 🤖 Available Models

### AC Model

Input:

* Non-corrected SPECT

Output:

* Attenuation-corrected SPECT

### SC Model

Input:

* Attenuation-corrected SPECT

Output:

* Monte Carlo-based scatter-corrected SPECT

### ASC Model

Input:

* Non-corrected SPECT

Output:

* Joint attenuation and scatter-corrected SPECT

---

## 📈 Performance Highlights

* SSIM > 99.8% across all correction tasks
* Gamma pass rates > 99.6%
* Strong agreement with reference Monte Carlo-based reconstructions
* Accurate voxel-wise and organ-level dosimetry

---

## 🔗 Related Work

This repository extends our previous work on quantitative 90Y-SPECT imaging:

**DL-based Attenuation and Scatter Correction for 90Y SPECT Imaging**

https://github.com/ZahraMansouriMedPhys/DL-90Y-SPECT-Correction

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{mansouri2026maa,
  title={Deep learning-guided attenuation and scatter correction of 99mTc-MAA SPECT images: towards quantitative analysis in 90Y-SIRT},
  author={Mansouri, Zahra and Salimi, Yazdan and Bianchetto Wolf, Nicola and Hajianfar, Ghasem and Mainta, Ismini and Garibotto, Valentina and Zaidi, Habib},
  journal={Annals of Nuclear Medicine},
  year={2026},
  doi={10.1007/s12149-025-02152-2}
}
```

---

## 🤝 Contributions

Feedback, suggestions, and contributions are welcome.

---

## 📧 Contact

**Zahra Mansouri**

📧 [zahra.mansouri@unige.ch](mailto:zahra.mansouri@unige.ch)
