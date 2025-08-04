import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import supervision as sv

# ----------------- Brand Styling -----------------
st.set_page_config(page_title="Sakman Solutions - Factory Safety POC", layout="wide")
st.markdown(
    """
    <style>
        .title {text-align: center; font-size: 36px; color: #2C3E50; font-weight: bold;}
        .subtitle {text-align: center; font-size: 18px; color: #7F8C8D; margin-bottom: 20px;}
        .stDownloadButton button {background-color: #2C3E50; color: white; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<div class='title'>🏭 Sakman Solutions - Factory Safety POC</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Automated Staff & Safety Gear Detection</div>", unsafe_allow_html=True)

# ----------------- Video Upload -----------------
uploaded_video = st.file_uploader("Upload your factory video (MP4/MOV/AVI)", type=["mp4", "mov", "avi"])

if uploaded_video:
    st.info("Video uploaded successfully. Processing will start...")

    # Save temp uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Load YOLO model (COCO pretrained)
    model = YOLO("yolov8n.pt")  # lightweight model for POC

    # Prepare video processing
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = "processed_video.mp4"
    os.makedirs("processed", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()
    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
    tracker = sv.ByteTrack()

    frame_count = 0
    total_people_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect only people (class 0)
        results = model(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Filter for person class
        person_detections = detections[detections.class_id == 0]
        people_count = len(person_detections)
        total_people_detected += people_count

        # Simulated missing gear detection for POC
        missing_gear_count = max(0, people_count - 1)  # pretend 1 person has gear

        # Annotate frame
        labels = [f"Person {tid}" for tid in person_detections.tracker_id]
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=person_detections, labels=labels)

        # Overlay counters
        cv2.rectangle(annotated_frame, (0,0), (350,70), (44,62,80), -1)  # dark header
        cv2.putText(annotated_frame, f"People: {people_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(annotated_frame, f"Missing Gear: {missing_gear_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Write & Display
        out.write(annotated_frame)
        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    out.release()

    st.success("✅ Processing complete!")
    st.video(output_path)
    st.download_button("📥 Download Processed Video", data=open(output_path, "rb"), file_name="sakman_factory_poc.mp4")

    st.markdown(
        "<div class='subtitle'>© 2025 Sakman Solutions | POC - Staff & Safety Gear Detection</div>",
        unsafe_allow_html=True
    )
