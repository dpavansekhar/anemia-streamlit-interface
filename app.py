import streamlit as st
from datetime import date
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import pandas as pd

# === Palm processing setup ===
PALM_MODEL_PATH = "models/mobilenet_model_palm.pth"
YOLO_MODEL_PATH = "models/best 10aug2025 eye.pt"
CLASSIFIER_MODEL_PATH = "models/MobileNetV2_eye.pth"
CLASS_TARGET = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths to cropped image folders ===
EYE_FOLDER = "cropped_data/Conjunctiva"
PALM_FOLDER = "cropped_data/palm"
NAIL_FOLDER = "cropped_data/Nail Beds"

def extract_palm(image_path, margin=20):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            landmark_ids = [0, 1, 5, 9, 13, 17]
            hand_landmarks = results.multi_hand_landmarks[0]
            coords = [(int(hand_landmarks.landmark[i].x * w),
                       int(hand_landmarks.landmark[i].y * h)) for i in landmark_ids]

            x_vals, y_vals = zip(*coords)
            x_min = max(min(x_vals) - margin, 0)
            x_max = min(max(x_vals) + margin, w)
            y_min = max(min(y_vals) - margin, 0)
            y_max = min(max(y_vals) + margin, h)

            palm_crop = image[y_min:y_max, x_min:x_max]
            return cv2.cvtColor(palm_crop, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"No hand detected in image: {image_path}")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(PALM_MODEL_PATH, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_palm(image_path):
    palm_rgb = extract_palm(image_path)
    palm_pil = Image.fromarray(palm_rgb)
    image_tensor = transform(palm_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        class_idx = torch.argmax(probs).item()
        confidence = float(probs[0][class_idx].item())

    prediction = "anemic" if class_idx == 1 else "non_anemic"
    return prediction, confidence

@st.cache_resource  # Cache so models load once
def load_models():
    yolo_model = YOLO(YOLO_MODEL_PATH)
    classifier_model = models.mobilenet_v2(weights=None)
    classifier_model.classifier = nn.Sequential(
        nn.Linear(classifier_model.last_channel, 1)
    )
    classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    classifier_model.to(DEVICE)
    classifier_model.eval()
    return yolo_model, classifier_model

yolo_model, classifier_model = load_models()

# Transform for classifier
eye_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def process_eye_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]

    results = yolo_model(image_path, imgsz=224, verbose=False)

    cropped_image = None
    for r in results:
        if r.masks is None:
            continue
        masks = r.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            cls = int(r.boxes.cls[i].cpu().numpy()) if len(r.boxes) > i else None
            if cls is not None and cls != CLASS_TARGET:
                continue

            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)

            nonz = cv2.findNonZero(mask_resized)
            if nonz is None:
                continue

            x, y, bw, bh = cv2.boundingRect(nonz)

            # Use mask to keep only conjunctiva area
            conjunctiva_only = cv2.bitwise_and(img, img, mask=mask_resized)

            # Crop tightly
            cropped_image = conjunctiva_only[y:y+bh, x:x+bw]

            # Convert BGR → RGB for correct colors
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            break
        if cropped_image is not None:
            break

    if cropped_image is None:
        raise ValueError("No conjunctiva mask detected.")

    cropped_pil = Image.fromarray(cropped_image)
    image_tensor = eye_transform(cropped_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = classifier_model(image_tensor)
        prob = torch.sigmoid(output).item()
        pred_class = 1 if prob > 0.5 else 0

    class_names = ["Non-Anemia", "Anemia"]
    prediction = class_names[pred_class]
    confidence = prob
    return cropped_image, prediction, confidence

# Excel file path
EXCEL_FILE = "patient_records.xlsx"

def display_cropped_images(employee_id):
    """Display cropped images for given employee ID"""
    st.subheader("Cropped Images:")

    cols = st.columns(3)

    # Eye Images
    with cols[0]:
        st.markdown("**Eye Images**")
        for side in ["right", "left"]:
            img_path = os.path.join(EYE_FOLDER, f"{employee_id}_{side}_eye.jpg")
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption=f"{side.capitalize()} Eye")

    # Palm Images
    with cols[1]:
        st.markdown("**Palm Images**")
        for side in ["right", "left"]:
            img_path = os.path.join(PALM_FOLDER, f"{employee_id}_{side}_palm.jpg")
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption=f"{side.capitalize()} Palm")

    # Nail Images
    with cols[2]:
        st.markdown("**Nail Images**")
        for side in range(1, 20):
            img_path = os.path.join(NAIL_FOLDER, f"{employee_id}_nail_{side}.jpg")
            if os.path.exists(img_path):
                st.image(
                    Image.open(img_path),
                    caption=f"Nail {side}"
                )

def save_patient_to_excel(patient_data):
    """Append patient data to Excel file"""
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        df = pd.concat([df, pd.DataFrame([patient_data])], ignore_index=True)
    else:
        df = pd.DataFrame([patient_data])
    df.to_excel(EXCEL_FILE, index=False)

# === Initialize page state ===
if "page" not in st.session_state:
    st.session_state.page = 1

# === PAGE 1: Patient Details Form ===
if st.session_state.page == 1:
    st.title("Patient Details Form")

    with st.form("patient_form"):
        st.subheader("Enter Patient Details")

        name = st.text_input("Name *")
        roll_id = st.text_input("Roll No. / Employee ID *")
        dob = st.date_input("Date of Birth *", min_value=date(1900, 1, 1), max_value=date.today())
        weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
        mobile = st.text_input("Mobile Number *")

        blood_group = st.selectbox(
            "Blood Group *",
            ["", "A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
        )

        role = st.radio("Role *", ["Student", "Faculty"])
        sex = st.radio("Sex *", ["Male", "Female", "Prefer Not to Say"])

        submitted = st.form_submit_button("Next ➡️")

    if submitted:
        missing_fields = []
        if not name.strip():
            missing_fields.append("Name")
        if not roll_id.strip():
            missing_fields.append("Roll No. / Employee ID")
        if not mobile.strip():
            missing_fields.append("Mobile Number")
        if blood_group.strip() == "":
            missing_fields.append("Blood Group")

        if missing_fields:
            st.error(f"Please fill in the required fields: {', '.join(missing_fields)}")
        else:
            st.session_state.patient_data = {
                "Name": name,
                "Roll/ID": roll_id,
                "DOB": str(dob),
                "Weight": weight,
                "Mobile": mobile,
                "Blood Group": blood_group,
                "Role": role,
                "Sex": sex
            }
            st.success("Details saved! Moving to the next page...")
            st.session_state.page = 2
            st.experimental_rerun()

# === PAGE 2: Palm Upload ===
elif st.session_state.page == 2:
    st.title("Palm Image Upload")

    if "patient_data" not in st.session_state:
        st.error("Please complete Page 1 first.")
        st.stop()

    employee_id = st.session_state.patient_data["Roll/ID"]  # ✅ match key with page 1
    INP_DIR = "uploaded_data/palm"
    CRP_DIR = "cropped_data/palm"
    os.makedirs(INP_DIR, exist_ok=True)
    os.makedirs(CRP_DIR, exist_ok=True)

    right_palm_file = st.file_uploader("Upload Right Palm Image", type=["jpg", "jpeg", "png"])
    left_palm_file = st.file_uploader("Upload Left Palm Image", type=["jpg", "jpeg", "png"])

    if st.button("Next ➡️"):
        if right_palm_file is None or left_palm_file is None:
            st.error("Please upload both right and left palm images.")
        else:
            right_path = os.path.join(INP_DIR, f"{employee_id}_right_palm.jpg")
            left_path = os.path.join(INP_DIR, f"{employee_id}_left_palm.jpg")

            with open(right_path, "wb") as f:
                f.write(right_palm_file.getbuffer())

            with open(left_path, "wb") as f:
                f.write(left_palm_file.getbuffer())

            try:
                # Extract cropped palms
                right_cropped_rgb = extract_palm(right_path)
                left_cropped_rgb = extract_palm(left_path)

                right_cropped_bgr = cv2.cvtColor(right_cropped_rgb, cv2.COLOR_RGB2BGR)
                left_cropped_bgr = cv2.cvtColor(left_cropped_rgb, cv2.COLOR_RGB2BGR)

                cropped_right_path = os.path.join(CRP_DIR, f"{employee_id}_right_palm.jpg")
                cropped_left_path = os.path.join(CRP_DIR, f"{employee_id}_left_palm.jpg")

                cv2.imwrite(cropped_right_path, right_cropped_bgr)
                cv2.imwrite(cropped_left_path, left_cropped_bgr)

                # Predictions
                right_pred, right_conf = predict_palm(right_path)
                left_pred, left_conf = predict_palm(left_path)

                # Combined prediction (majority vote)
                preds = [right_pred, left_pred]
                if preds.count("Anemia") > preds.count("Non-Anemia"):
                    combined_pred = "Anemia"
                else:
                    combined_pred = "Non-Anemia"

                avg_conf = round((right_conf + left_conf) / 2, 4)

                # Save results
                st.session_state.palm_results = {
                    "right": {"prediction": right_pred, "confidence": right_conf},
                    "left": {"prediction": left_pred, "confidence": left_conf},
                    "combined_prediction": combined_pred,
                    "average_confidence": avg_conf
                }

                st.success(f"Palms processed!!")
                st.session_state.page = 3
                st.experimental_rerun()

            except Exception as e:
                st.error(f"Error processing palms: {e}")

# === PAGE 3: (Optional) Confirmation or next steps ===
elif st.session_state.page == 3:
    st.title("Eye Image Upload")

    if "patient_data" not in st.session_state:
        st.error("Please complete previous steps first.")
        st.stop()

    employee_id = st.session_state.patient_data["Roll/ID"]  # Make sure key matches your page 1 storage
    EYE_INP_DIR = "uploaded_data/Conjunctiva"
    EYE_CRP_DIR = "cropped_data/Conjunctiva"
    os.makedirs(EYE_INP_DIR, exist_ok=True)
    os.makedirs(EYE_CRP_DIR, exist_ok=True)

    right_eye_file = st.file_uploader("Upload Right Eye Image", type=["jpg", "jpeg", "png"])
    left_eye_file = st.file_uploader("Upload Left Eye Image", type=["jpg", "jpeg", "png"])

    if st.button("Next ➡️"):
        if right_eye_file is None or left_eye_file is None:
            st.error("Please upload both right and left eye images.")
        else:
            right_eye_path = os.path.join(EYE_INP_DIR, f"{employee_id}_right_eye.jpg")
            left_eye_path = os.path.join(EYE_INP_DIR, f"{employee_id}_left_eye.jpg")

            with open(right_eye_path, "wb") as f:
                f.write(right_eye_file.getbuffer())

            with open(left_eye_path, "wb") as f:
                f.write(left_eye_file.getbuffer())

            try:
                # Process right eye
                right_cropped, right_pred, right_conf = process_eye_image(right_eye_path)
                cropped_right_path = os.path.join(EYE_CRP_DIR, f"{employee_id}_right_eye.jpg")
                cv2.imwrite(cropped_right_path, cv2.cvtColor(right_cropped, cv2.COLOR_RGB2BGR))

                # Process left eye
                left_cropped, left_pred, left_conf = process_eye_image(left_eye_path)
                cropped_left_path = os.path.join(EYE_CRP_DIR, f"{employee_id}_left_eye.jpg")
                cv2.imwrite(cropped_left_path, cv2.cvtColor(left_cropped, cv2.COLOR_RGB2BGR))

                # Determine combined prediction
                preds = [right_pred, left_pred]
                if preds.count("Anemia") > preds.count("Non-Anemia"):
                    combined_pred = "Anemia"
                else:
                    combined_pred = "Non-Anemia"

                avg_conf = round((right_conf + left_conf) / 2, 4)

                # Store results in session
                st.session_state.eye_results = {
                    "right": {"prediction": right_pred, "confidence": right_conf},
                    "left": {"prediction": left_pred, "confidence": left_conf},
                    "combined_prediction": combined_pred,
                    "average_confidence": avg_conf
                }

                st.success(f"Conjunctiva Images processed!!")
                st.session_state.page = 4
                st.experimental_rerun()

            except Exception as e:
                st.error(f"Error processing eye images: {e}")

elif st.session_state.page == 4:
    st.title("Nail Bed Image Upload")

    if "patient_data" not in st.session_state:
        st.error("Please complete previous steps first.")
        st.stop()

    employee_id = st.session_state.patient_data["Roll/ID"]  # ✅ match page 1 key

    # === Folders ===
    NAIL_INPUT_DIR = "uploaded_data/Nail Beds"
    NAIL_CROP_DIR = "cropped_data/Nail Beds"
    os.makedirs(NAIL_INPUT_DIR, exist_ok=True)
    os.makedirs(NAIL_CROP_DIR, exist_ok=True)

    # === Load YOLO & CNN models (cached) ===
    @st.cache_resource
    def load_nail_models():
        yolo_model = YOLO("models/yolov8_nail_best.pt")

        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=2):
                super(SimpleCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, 1, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(16, 32, 3, 1, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 56 * 56, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        anemia_model = SimpleCNN()
        anemia_model.load_state_dict(torch.load("models/anemia_cnn_model2_nail.pth", map_location=DEVICE))
        anemia_model.to(DEVICE)
        anemia_model.eval()

        return yolo_model, anemia_model

    yolo_model, anemia_model = load_nail_models()

    # Transform for CNN
    anemia_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # === Upload UI ===
    nail_image_file = st.file_uploader("Upload hand image showing nails", type=["jpg", "jpeg", "png"])

    if st.button("Next ➡️"):
        if nail_image_file is None:
            st.error("Please upload a nail bed image.")
        else:
            # Save input image
            input_path = os.path.join(NAIL_INPUT_DIR, f"{employee_id}_nail.jpg")
            with open(input_path, "wb") as f:
                f.write(nail_image_file.getbuffer())

            try:
                # Run YOLO detection
                image_pil = Image.open(input_path).convert("RGB")
                results = yolo_model(image_pil)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                predictions = []
                confidences = []

                # Process each detected nail
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    nail_crop = image_pil.crop((x1, y1, x2, y2))

                    # Save cropped nail
                    crop_path = os.path.join(NAIL_CROP_DIR, f"{employee_id}_nail_{i+1}.jpg")
                    nail_crop.save(crop_path)

                    # CNN classification
                    nail_tensor = anemia_transform(nail_crop).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        output = anemia_model(nail_tensor)
                        probs = torch.softmax(output, dim=1)
                        anemia_class = torch.argmax(probs).item()
                        confidence = probs[0][anemia_class].item()

                    # Convert class index to label
                    label = "Anemia" if anemia_class == 1 else "Non-Anemia"
                    predictions.append(label)
                    confidences.append(confidence)

                # Final aggregation
                if predictions:
                    combined_pred = "Anemia" if predictions.count("Anemia") > predictions.count("Non-Anemia") else "Non-Anemia"
                    avg_conf = round(sum(confidences) / len(confidences), 4)
                else:
                    combined_pred = "Non-Anemia"
                    avg_conf = 0.0

                st.session_state.nail_results = {
                    "per_nail": [{"prediction": p, "confidence": c} for p, c in zip(predictions, confidences)],
                    "combined_prediction": combined_pred,
                    "average_confidence": avg_conf
                }

                st.success(f"Nail Beds processed!!")
                st.session_state.page = 5
                st.experimental_rerun()

            except Exception as e:
                st.error(f"Error processing nails: {e}")

# ===== PAGE 5 CODE =====
if st.session_state.page == 5:
    st.title("Summary")

    # Show patient details
    st.subheader("Patient Details:")
    import pandas as pd
    patient_df = pd.DataFrame(
        list(st.session_state.patient_data.items()),
        columns=["Field", "Value"]
    )
    st.table(patient_df)

    # Display cropped images
    display_cropped_images(st.session_state.patient_data["Roll/ID"])

    # Show predictions
    st.subheader("Predictions:")
    combined_eye = st.session_state.eye_results.get("combined_prediction", "N/A")
    combined_palm = st.session_state.palm_results.get("combined_prediction", "N/A")
    combined_nail = st.session_state.nail_results.get("combined_prediction", "N/A")

    # Simple logic for overall prediction (can replace with your actual logic)
    predictions_list = [combined_eye, combined_palm, combined_nail]
    overall_prediction = max(set(predictions_list), key=predictions_list.count)  # majority voting

    st.write(f"**Combined Eye Prediction:** {combined_eye}")
    st.write(f"**Combined Palm Prediction:** {combined_palm}")
    st.write(f"**Combined Nail Prediction:** {combined_nail}")
    st.write(f"**Overall Prediction:** {overall_prediction}")

    # Optional HB value input
    hb_value = st.number_input("Enter Hb Value (optional)", min_value=0.0, step=0.1, format="%.1f")

    # Save patient data to Excel
    if st.button("Save Record"):
        save_data = {
            **st.session_state.patient_data,
            "Hb Value": hb_value if hb_value else None,
            "Combined Eye Prediction": combined_eye,
            "Combined Palm Prediction": combined_palm,
            "Combined Nail Prediction": combined_nail,
            "Overall Prediction": overall_prediction
        }
        save_patient_to_excel(save_data)
        st.success("Patient data saved successfully!")

    # Restart session
    if st.button("Restart"):
        for key in ["page", "patient_data", "eye_results", "palm_results", "nail_results"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()
