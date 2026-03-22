import streamlit as st
import cv2
import numpy as np
import base64
import tempfile
import os
from gtts import gTTS
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Human Facial Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }

    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #00d9ff, #00ff88, #00d9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .result-box {
        background: linear-gradient(145deg, rgba(0, 217, 255, 0.1), rgba(0, 255, 136, 0.05));
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .emotion-label {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00ff88;
    }

    .confidence-label {
        font-size: 1.1rem;
        color: #00d9ff;
    }

    .error-msg {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 8px;
        padding: 1rem;
        color: #f87171;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants
# -----------------------------
MODEL_PATH = "emotion_model.keras"
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -----------------------------
# Load model + cascade once
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return model, face_cascade


def preprocess_face(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


def detect_emotions_in_image(image):
    model, face_cascade = load_resources()

    result_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    predictions = []

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        if face_gray.size == 0:
            continue

        processed = preprocess_face(face_gray)
        prediction = model.predict(processed, verbose=0)[0]
        class_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        emotion = CLASS_NAMES[class_idx]

        predictions.append({
            "box": (x, y, w, h),
            "emotion": emotion,
            "confidence": confidence
        })

        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            result_img,
            f"{emotion} ({confidence:.1%})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return result_img, predictions


def process_image_for_emotion(image: np.ndarray):
    try:
        result_img, predictions = detect_emotions_in_image(image)

        if not predictions:
            return None, [], []

        emotions = [pred["emotion"] for pred in predictions]
        confidences = [pred["confidence"] for pred in predictions]

        return result_img, emotions, confidences
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, [], []


def get_emotion_message(emotion):
    emotion = str(emotion).strip().lower()
    messages = {
        "happy": "You are looking very happy",
        "sad": "You are looking sad",
        "angry": "You are looking angry",
        "surprise": "You are looking surprised",
        "fear": "You are looking fearful",
        "disgust": "You are looking disgusted",
        "neutral": "You are looking neutral"
    }
    return messages.get(emotion, f"Detected emotion is {emotion}")


def generate_tts_audio_bytes(text):
    tts = gTTS(text=text, lang="en")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name

    tts.save(temp_path)

    with open(temp_path, "rb") as f:
        audio_bytes = f.read()

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return audio_bytes


def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay="true" controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


def display_results(result_img, emotions, confidences, enable_voice=True):
    st.subheader("Result")
    st.image(result_img, channels="BGR", use_container_width=True)

    for emotion, conf in zip(emotions, confidences):
        st.markdown(
            f"""
            <div class="result-box">
                <div class="emotion-label">Emotion: {emotion}</div>
                <div class="confidence-label">Confidence: {conf:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if enable_voice and emotions:
        primary_emotion = emotions[0]
        speech_text = get_emotion_message(primary_emotion)

        st.markdown("### Voice Output")
        st.write(f"Speaking: **{speech_text}**")

        try:
            audio_bytes = generate_tts_audio_bytes(speech_text)
            autoplay_audio(audio_bytes)
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.warning(f"Could not generate voice output: {e}")
            st.info("gTTS needs internet connection.")


# -----------------------------
# WebRTC processor for realtime video
# -----------------------------
class EmotionVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result_img, _ = detect_emotions_in_image(img)
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")


def main():
    st.title("Human Facial Emotion Detection System")
    st.markdown(
        '<p class="subtitle">Upload an image or use real-time webcam detection in the browser.</p>',
        unsafe_allow_html=True,
    )

    st.sidebar.header("Options")
    mode = st.sidebar.radio(
        "Select Mode",
        ["Upload Image", "Realtime Webcam"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Emotions Detected")
    st.sidebar.markdown("Angry • Disgust • Fear • Happy • Neutral • Sad • Surprise")

    if mode == "Upload Image":
        render_upload_mode()
    else:
        render_realtime_webcam_mode()


def render_upload_mode():
    st.header("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a photo containing a face to detect emotions",
    )

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Invalid image file. Please upload a valid image.")
                return

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Input Image")
                st.image(image, channels="BGR", use_container_width=True)

            with st.spinner("Detecting faces and emotions..."):
                result_img, emotions, confidences = process_image_for_emotion(image)

            with col2:
                if result_img is not None:
                    display_results(result_img, emotions, confidences, enable_voice=True)
                else:
                    st.markdown(
                        """
                        <div class="error-msg">
                            No face detected in the image. Please ensure the image contains a clear, visible face.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"Error processing image: {e}")


def render_realtime_webcam_mode():
    st.header("Realtime Webcam Detection")
    st.markdown(
        "Allow camera access in your browser, then click **START** on the webcam component."
    )

    webrtc_streamer(
        key="emotion-realtime",
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )


if __name__ == "__main__":
    main()