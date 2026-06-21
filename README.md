# Medical Image Classification — AneRBC Dataset

Custom CNNs (3/4/5 layers), Transfer Learning (MobileNetV2, ResNet18, DenseNet121), and Explainable AI (Grad-CAM) for blood smear anemia classification (Anemic vs Healthy).

**Name:** <NIMRA SAEED>
**SRN:** <303231009>

---

## 1. Environment Setup

This project was developed and run on **Google Colab**.

### Requirements
- Python 3.10+
- PyTorch + torchvision
- OpenCV (`opencv-python`)
- numpy, pandas, matplotlib
- scikit-learn
- `grad-cam` (Captum-style Grad-CAM implementation for PyTorch)

### Installing dependencies
If running outside Colab (e.g. locally or on Kaggle), install with:

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib scikit-learn grad-cam
```

On Colab, only the XAI library needs manual installation (everything else is preinstalled):

```python
!pip install grad-cam -q
```

### GitHub authentication (for contributors)
Do **not** hardcode a GitHub token in any notebook cell. Use Colab's Secrets manager:
1. Click the 🔑 key icon in the left sidebar of Colab
2. Add a secret named `GITHUB_PAT` with your personal access token as the value
3. Enable notebook access for the secret

The setup cell reads it via:
```python
from google.colab import userdata
GITHUB_PAT = userdata.get('GITHUB_PAT')
```

---

## 2. Dataset Setup

**Dataset:** AneRBC (Macroscopic Blood Smear Cell Image Dataset)
**Classes:** Anemic (0), Healthy (1)

### Download / prepare
1. Download the AneRBC dataset from its original source (see report for citation/link).
2. Place it in your Google Drive under:
   ```
   MyDrive/anerbc_dataset/AneRBC dataset/AneRBC_dataset/
   ├── AneRBC-I/
   │   ├── Anemic_individuals/Original_images/
   │   └── Healthy_individuals/Original_images/
   └── AneRBC-II/
       ├── Anemic_individuals/Original_images/
       └── Healthy_individuals/Original_images/
   ```
3. Mount Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Update `BASE_PATH` in the notebook if your folder name/path differs.

The dataset itself is **not stored in this Git repository** (excluded via `.gitignore`) — only code, trained model checkpoints, and XAI outputs are version-controlled.

---

## 3. Project Structure

```
.
├── ANN_DL_PRJ_Solution.ipynb   # Main notebook (Tasks 1-4)
├── checkpoints/                 # Saved model weights (.pt) and Grad-CAM outputs (.png)
├── README.md
└── .gitignore
```

---

## 4. How to Train Each Model Family

Open `ANN_DL_PRJ_Solution.ipynb` in Google Colab and run cells **top to bottom**.

### Task 1 — Data preparation
Runs validation (corruption checks, class distribution), preprocessing (resize to 128x128, ImageNet normalization), and a stratified train/val/test split (seed=42).

### Task 2 — Custom CNNs
Trains three custom architectures (`CNN3Layer`, `CNN4Layer`, `CNN5Layer`) for 10 epochs each (Adam optimizer, lr=1e-3, CrossEntropyLoss). Each model's weights are saved to `checkpoints/<name>.pt` after training.

```python
results = {}
for name, ModelClass in [("CNN3", CNN3Layer), ("CNN4", CNN4Layer), ("CNN5", CNN5Layer)]:
    model = ModelClass()
    model, history = train_model(model, train_loader, val_loader, epochs=10)
    ...
```

### Task 3 — Transfer learning
Fine-tunes the classifier head of three pretrained backbones (MobileNetV2, ResNet18, DenseNet121) with the backbone frozen. Trainable layers are documented inline in `build_transfer_model()`.

```python
for name in ["mobilenet", "resnet", "densenet"]:
    tmodel = build_transfer_model(name)
    tmodel, hist = train_model(tmodel, train_loader, val_loader, epochs=5)
    ...
```

**Note:** training time is significantly faster with a GPU runtime (Runtime → Change runtime type → GPU). On CPU, expect substantially longer per-epoch times, especially for Task 3.

---

## 5. How to Run Evaluation and Generate XAI Outputs

### Evaluation
`evaluate_model()` is called automatically after training each model in Tasks 2 and 3. It prints accuracy, a full classification report (precision/recall/F1), and plots a confusion matrix for the held-out test set.

### XAI (Grad-CAM)
Run the Task 4 cell to generate Grad-CAM heatmaps for the best-performing custom CNN and the best-performing pretrained model:

```python
cam_custom = run_gradcam(best_custom_model, best_custom_model.features[-3], inp, raw_resized)
cam_pretrained = run_gradcam(best_pretrained_model, target_layer, inp, raw_resized)
```

Outputs are saved as `checkpoints/gradcam_custom.png` and `checkpoints/gradcam_pretrained.png`.

---

## 6. Reloading Saved Checkpoints (no retraining needed)

If returning to this project after a runtime disconnect, model weights can be reloaded instead of retraining:

```python
model = CNN3Layer()
model.load_state_dict(torch.load("checkpoints/CNN3.pt", map_location=device))
model = model.to(device)
```

---

## 7. Git Workflow

Commits were made after completing each subtask, per the assignment's strict commit policy:

```bash
git add .
git commit -m "TaskX.Y: <description>"
git push
```

See commit history for the full task-by-task progression.

---

## 8. Report

The critical evaluation report (≤5000 words) covering dataset description, custom vs pretrained model comparison, and XAI-based critique is submitted separately to the University LMS, as required by the assignment brief.
