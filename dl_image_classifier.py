import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import cv2

# -------------------------------
# Cache resource for model
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

# -------------------------------
# Cache resource for ImageNet labels
# -------------------------------
@st.cache_resource
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            labels.append(line.decode("utf-8").strip())
    return labels

# -------------------------------
# Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Grad-CAM function
# -------------------------------
def generate_gradcam(model, image_tensor, target_class):
    """Generate Grad-CAM heatmap for a given class"""
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register hooks on last conv layer
    layer = model.layer4[-1].conv2
    handle_fw = layer.register_forward_hook(forward_hook)
    handle_bw = layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()

    grads = gradients[0][0]          # C x H x W
    acts = activations[0][0]         # C x H x W

    weights = grads.mean(dim=(1, 2)) # Global average pooling
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.numpy()
    cam = cv2.resize(cam, (224, 224))

    handle_fw.remove()
    handle_bw.remove()
    return cam

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🧠 Deep Learning Image Classifier with Interpretability")
st.write("Upload an image, get predictions, confidence graphs, and Grad-CAM heatmaps.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    x = transform(img).unsqueeze(0)

    # Load model and labels
    model = load_model()
    labels = load_labels()

    # Predict
    with torch.no_grad():
        preds = model(x)
        probs = torch.nn.functional.softmax(preds[0], dim=0)
    
    # Top-5 predictions
    top5_prob, top5_idx = torch.topk(probs, 5)

    st.subheader("🔮 Top-5 Predictions")
    for i in range(top5_prob.size(0)):
        st.write(f"{labels[top5_idx[i]]}: **{top5_prob[i].item()*100:.2f}%**")

    # Bar chart of top-5 probabilities
    st.subheader("📊 Confidence Plot")
    import plotly.express as px
    top_labels = [labels[i] for i in top5_idx]
    fig = px.bar(x=top_labels, y=top5_prob.numpy()*100, labels={"x":"Class", "y":"Confidence (%)"}, title="Top-5 Prediction Confidence")
    st.plotly_chart(fig)

    # Grad-CAM for top prediction
    st.subheader("🔥 Grad-CAM Interpretability")
    target_class = top5_idx[0].item()
    cam = generate_gradcam(model, x, target_class)

    # Overlay CAM on original image
    img_resized = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption=f"Grad-CAM for '{labels[target_class]}'", use_container_width=True)
