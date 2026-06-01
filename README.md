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

The trained model weights are available upon request through the following Google Drive folder:

https://drive.google.com/drive/folders/1kBq7gA8YaatLOVIYUyv6eIHMwV7BhWGx?usp=drive_link

Access is managed by the authors. Please request access through Google Drive and briefly indicate your institution and intended use.

Available models:

* **AC Model** — CT-free attenuation correction
* **SC Model** — Monte Carlo-based scatter correction
* **ASC Model** — Joint attenuation and scatter correction

---

## 🚀 Installation

```bash
pip install git+https://github.com/ZahraMansouriMedPhys/DL-99mTc-MAA-SPECT-Correction.git
```

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
