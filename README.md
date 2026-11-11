# Brain MRI Tumor Classifier (Junior CV Project)

**Stack:** Python, OpenCV, scikit-image (HOG), scikit-learn (SVM), Matplotlib.  
**Goal:** Binary *image-level* classification: `tumor` vs. `no tumor`. It does **not** locate, segment, subtype, grade, or medically diagnose a tumor.

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Get a **lawfully usable** labeled brain MRI dataset and arrange its images as follows:

```text
data/
  yes/    # images labeled tumor
  no/     # images labeled no tumor
```

Suggested search: a labeled binary brain MRI image dataset with `yes` / `no` folders. Images are deliberately **not included**. Check the source's license and provenance before using it.

```powershell
python train.py
python predict.py --image data/yes/example.jpg
```

Outputs: `models/brain_mri_model.joblib`, `outputs/confusion_matrix.png`, `outputs/metrics.csv`.

An optional step-by-step notebook, `Brain_MRI_Classifier.ipynb`, mirrors the same workflow for Jupyter / VS Code.

## Pipeline

1. Read MRI images using OpenCV and convert to grayscale.
2. Resize to 128x128, improve local contrast with CLAHE.
3. Extract HOG features (image gradients and shapes).
4. Split train/test (stratified) and tune SVM with GridSearchCV **on the training split**.
5. Report test accuracy, precision, recall, F1 and confusion matrix.
6. Save model and predict one image.

## Important limitations

- **For portfolio and learning only. Never use for patient care.** False negatives are possible.
- A random image split can seriously overestimate performance if the same patient has several images. For real evaluation, use patient-level train/test groups and an independent external dataset.
- Check dataset composition, scanner/site differences, label quality and licensing; address class imbalance and bias.
- `scoring='f1'` is used here because the labels are numeric 0/1 and both precision and recall matter. High test accuracy alone does not establish medical safety.
- No performance claim is made until training on actual labeled data.
