# ============================================================
# 🧬 Hybrid Quantum-Inspired Autism MRI Classifier (GPU)
# Author: Shrivatsa Gudi
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import nibabel as nib
from PIL import Image
from torchvision import models, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Hybrid Quantum MRI Classifier", page_icon="🧠", layout="wide")

st.title("🧬 Hybrid Quantum-Inspired MRI Classifier (CUDA-Powered)")
st.markdown("Upload an MRI scan (.nii or .png) to classify between **Autistic** and **Typical Control** brains.")
st.markdown("---")

# ============================================================
# 1️⃣ Load Scaler, PCA, and Model
# ============================================================
try:
    scaler = joblib.load("saved_models/scaler.pkl")
    pca = joblib.load("saved_models/pca_model.pkl")
except Exception as e:
    st.error(f"❌ Missing scaler or PCA file: {e}")
    st.stop()

# ============================================================
# 2️⃣ Define Model Architecture (Same as Training)
# ============================================================
class QuantumLikeLayer(nn.Module):
    def __init__(self, in_features, n_qubits=6, n_layers=12):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 3, device=DEVICE) * 0.2)
        self.linear_map = nn.Linear(in_features, n_qubits).to(DEVICE)
    def forward(self, x):
        x = torch.tanh(self.linear_map(x)) * np.pi
        for l in range(self.n_layers):
            sin_term = torch.sin(x + self.theta[l,:,0])
            cos_term = torch.cos(x + self.theta[l,:,1])
            x = sin_term + cos_term + torch.sin(x * self.theta[l,:,2])
            x = torch.matmul(x, torch.ones((self.n_qubits, self.n_qubits), device=DEVICE)) / self.n_qubits
        return x

class HybridVQC_CUDA(nn.Module):
    def __init__(self, in_features, n_qubits=6, n_layers=12):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, in_features),
            nn.ReLU()
        ).to(DEVICE)
        self.q_layer = QuantumLikeLayer(in_features, n_qubits, n_layers).to(DEVICE)
        self.fc_out = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(16, 2)
        ).to(DEVICE)
    def forward(self, x):
        x = self.fc_in(x)
        q_out = self.q_layer(x)
        return self.fc_out(q_out)

# ============================================================
# 3️⃣ Load Trained Model Weights
# ============================================================
MODEL_PATH = "saved_models/hybrid_vqc_final.pth"
try:
    model = HybridVQC_CUDA(32, n_qubits=6, n_layers=12).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    st.success("✅ Model loaded successfully on CUDA.")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

# ============================================================
# 4️⃣ Define Feature Extractor (ResNet18)
# ============================================================
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ============================================================
# 5️⃣ Helper: Load NIfTI slice
# ============================================================
def nii_to_pil_slices(nii_file):
    # Save uploaded Streamlit file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(nii_file.read())
        tmp_path = tmp.name

    # Load the NIfTI image from the saved temp file
    img = nib.load(tmp_path).get_fdata()
    z = img.shape[2] // 2
    slice_ = img[:, :, z]
    slice_ = (255 * (slice_ - np.min(slice_)) / np.ptp(slice_)).astype(np.uint8)
    return Image.fromarray(slice_).convert("RGB")

# ============================================================
# 6️⃣ File Upload Interface
# ============================================================
uploaded = st.file_uploader("📤 Upload MRI file (.nii, .nii.gz, .png, .jpg, .jpeg)", type=["nii", "nii.gz", "png", "jpg", "jpeg"])

if uploaded:
    file_ext = Path(uploaded.name).suffix
    if "nii" in file_ext:
        img = nii_to_pil_slices(uploaded)
    else:
        img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="Uploaded MRI Slice", width=300)

    # ============================================================
    # 7️⃣ Feature Extraction -> Scaling -> PCA -> Model Prediction
    # ============================================================
    with torch.no_grad():
        xb = transform(img).unsqueeze(0).to(DEVICE)
        feat = resnet(xb).cpu().numpy()
        feat_scaled = scaler.transform(feat)
        feat_pca = pca.transform(feat_scaled)
        feat_tensor = torch.tensor(feat_pca, dtype=torch.float32).to(DEVICE)
        logits = model(feat_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)

    classes = ["Typical Control", "Autistic"]
    pred_class = classes[pred]

    st.markdown("---")
    st.subheader(f"🧠 Prediction: **{pred_class}**")
    st.write(f"Confidence — Typical: `{probs[0]*100:.2f}%`, Autistic: `{probs[1]*100:.2f}%`")

    # ============================================================
    # 8️⃣ Visualize Confidence
    # ============================================================
    st.markdown("### 📊 Confidence Levels")
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.bar(classes, probs, color=["#66CCFF", "#FF9999"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # ============================================================
    # 9️⃣ Optional: Download Prediction Result
    # ============================================================
    result_str = f"Prediction: {pred_class}\nConfidence: Typical={probs[0]*100:.2f}%, Autistic={probs[1]*100:.2f}%"
    st.download_button("💾 Download Prediction Result", result_str, file_name="prediction.txt")

else:
    st.info("👆 Upload a .nii or .png file to begin classification.")
