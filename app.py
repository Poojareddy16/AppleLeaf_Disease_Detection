import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model_arch import PlantDiseaseModel
import io

st.title("🍎 Apple Leaf Disease Classifier")
st.markdown("Upload your trained model file (`.pth`) and an image to detect the disease.")

model_file = st.file_uploader("Upload `best_model.pth`", type=["pth"])
if model_file is not None:
    buffer = io.BytesIO(model_file.read())

    model = PlantDiseaseModel(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
    model.eval()
    st.success("✅ Model loaded successfully!")

    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']

    img_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0).numpy()
            pred_class = classes[np.argmax(probs)]

        st.subheader(f"🧠 Predicted Class: `{pred_class}`")
        st.bar_chart(dict(zip(classes, probs)))