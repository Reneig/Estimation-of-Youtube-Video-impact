import streamlit as st
import os
import subprocess
import json
import cv2
import tempfile
import glob
import numpy as np
import pandas as pd
import librosa
import io
from google.cloud import videointelligence_v1 as videointelligence
from google.oauth2 import service_account
import google.generativeai as genai
from openai import OpenAI
from IPython.display import Markdown, display, update_display

# ============================
# CONFIGURATION
# ============================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

GOOGLE_KEY_PATH = "civil-oarlock-476523-r4-4a3bf0be8497.json"
GEMINI_API_KEY = "AIzaSyD9EOE58SYhNkKLnidaw-RK_8ePKJ_q548"
OPENAI_API_KEY = "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"

FFMPEG_PATH = r"C:\Users\gbodo\PycharmProjects\PythonProject3\video_analyzer_app\ffmpeg\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\Users\gbodo\PycharmProjects\PythonProject3\video_analyzer_app\ffmpeg\bin\ffprobe.exe"

# ============================
# FONCTIONS UTILITAIRES
# ============================

def download_video(url):
    """Télécharge la vidéo YouTube et renvoie le chemin local + métadonnées"""
    try:
        result = subprocess.run(['yt-dlp', '-j', '--no-playlist', url], capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout.strip().splitlines()[0])
        video_title = metadata.get('title', 'video').replace('/', '').replace('\\', '')
        video_description = metadata.get('description', '')

        output_template = os.path.join(VIDEO_DIR, f"{video_title}.%(ext)s")
        subprocess.run(['yt-dlp', '-f', 'bestvideo+bestaudio/best', '-o', output_template, url], check=True)

        for file in os.listdir(VIDEO_DIR):
            if file.startswith(video_title) and file.split('.')[-1] in ['mp4', 'webm', 'mkv']:
                metadata_clean = {
                    "title": video_title,
                    "description": video_description
                }
                return os.path.join(VIDEO_DIR, file), metadata_clean

        return None, {"title": video_title, "description": video_description}
    except Exception as e:
        st.error(f"Erreur lors du téléchargement : {e}")
        return None, {}

def save_uploaded_video(uploaded_file):
    """Enregistre et convertit la vidéo uploadée localement (.mp4, .m4a, .webm)"""
    try:
        file_name = os.path.splitext(uploaded_file.name)[0]
        input_path = os.path.join(VIDEO_DIR, uploaded_file.name)

        # Enregistrement brut
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Conversion en .mp4, .m4a, .webm
        mp4_path = os.path.join(VIDEO_DIR, f"{file_name}.mp4")
        m4a_path = os.path.join(VIDEO_DIR, f"{file_name}.m4a")
        webm_path = os.path.join(VIDEO_DIR, f"{file_name}.webm")

        subprocess.run([FFMPEG_PATH, "-y", "-i", input_path, "-c:v", "libx264", "-c:a", "aac", mp4_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([FFMPEG_PATH, "-y", "-i", input_path, "-vn", "-acodec", "aac", m4a_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([FFMPEG_PATH, "-y", "-i", input_path, "-c:v", "libvpx-vp9", "-c:a", "libopus", webm_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        metadata = {"title": file_name, "description": "Vidéo téléversée par l’utilisateur."}
        st.success("✅ Vidéo téléversée et convertie avec succès !")
        return mp4_path, metadata

    except Exception as e:
        st.error(f"Erreur lors de la conversion : {e}")
        return None, {}

def get_video_metadata(video_path):
    """Récupère les métadonnées techniques via FFprobe"""
    try:
        cmd = [FFPROBE_PATH, "-v", "error",
               "-show_entries", "format=duration:stream=codec_name,codec_type,width,height",
               "-of", "json", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        duration = float(data["format"].get("duration", 0))
        video_codec, audio_codec, width, height = None, None, None, None

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_codec = stream.get("codec_name")
                width = stream.get("width")
                height = stream.get("height")
            elif stream.get("codec_type") == "audio":
                audio_codec = stream.get("codec_name")

        return {
            "duration_sec": round(duration, 2),
            "video_codec": video_codec,
            "audio_codec": audio_codec,
            "width": width,
            "height": height
        }
    except Exception as e:
        st.error(f"Erreur métadonnées : {e}")
        return {}

def analyze_video_with_key(video_path, key_path=GOOGLE_KEY_PATH):
    """Analyse vidéo avec Google Video Intelligence (équivalent code Colab)"""
    try:
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)
        features = [
            videointelligence.Feature.SHOT_CHANGE_DETECTION,
            videointelligence.Feature.OBJECT_TRACKING
        ]

        with io.open(video_path, "rb") as f:
            input_content = f.read()

        request = {"input_content": input_content, "features": features}
        operation = client.annotate_video(request=request)
        st.info("Analyse Google Video Intelligence en cours... (~2-4 min)")
        result = operation.result(timeout=900)

        annotation_result = result.annotation_results[0]
        out = {}

        # Nombre de changements de plans
        out["shot_changes_count"] = len(annotation_result.shot_annotations or [])

        # Objets détectés (max 50)
        objects = []
        for obj in (annotation_result.object_annotations or [])[:50]:
            desc = obj.entity.description
            conf = getattr(obj, "confidence", None)
            seg_start = getattr(obj.segment.start_time_offset, "seconds", 0)
            seg_end = getattr(obj.segment.end_time_offset, "seconds", 0)
            objects.append({"description": desc, "confidence": conf, "start_s": seg_start, "end_s": seg_end})

        out["object_tracking_count"] = len(objects)
        return out

    except Exception as e:
        st.error(f"Erreur Google Video Intelligence : {e}")
        return {"shot_changes_count": 0, "object_tracking_count": 0}

def analyze_visual_quality(video_path, sample_frames=10):
    """Analyse visuelle : luminosité, contraste, netteté, couleur dominante"""
    try:
        temp_dir = tempfile.mkdtemp()
        duration = librosa.get_duration(filename=video_path)
        cmd = [
            FFMPEG_PATH, "-i", video_path,
            "-vf", f"fps=1/{max(1, int(duration // sample_frames))}",
            os.path.join(temp_dir, "frame_%03d.jpg"),
            "-hide_banner", "-loglevel", "error"
        ]
        subprocess.run(cmd, check=True)
        image_files = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))

        if not image_files:
            return {"mean_brightness": 0, "mean_contrast": 0, "mean_sharpness": 0,
                    "dominant_color_rgb": (0, 0, 0), "dominant_color_hex": "#000000"}

        brightness, contrast, sharpness, colors = [], [], [], []
        for img_path in image_files:
            frame = cv2.imread(img_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(gray.mean())
            contrast.append(gray.std())
            sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            data = cv2.resize(frame, (100, 100)).reshape(-1, 3)
            colors.append(np.mean(data, axis=0))

        mean_color = np.mean(np.array(colors), axis=0)
        hex_color = '#%02x%02x%02x' % tuple(mean_color.astype(int))

        return {
            "mean_brightness": round(np.mean(brightness), 2),
            "mean_contrast": round(np.mean(contrast), 2),
            "mean_sharpness": round(np.mean(sharpness), 2),
            "dominant_color_rgb": tuple(mean_color.astype(int)),
            "dominant_color_hex": hex_color
        }
    except Exception as e:
        st.error(f"Erreur analyse visuelle : {e}")
        return {}

def analyze_audio(video_path):
    """Analyse audio basique avec Librosa"""
    try:
        audio_path = "temp_audio.wav"
        subprocess.run([FFMPEG_PATH, "-y", "-i", video_path, "-vn",
                        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        y, sr = librosa.load(audio_path, sr=None)
        os.remove(audio_path)
        return {
            "rms_volume": float(np.mean(librosa.feature.rms(y=y))),
            "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            "noise_level": float(np.mean(np.abs(y[y < 0.01])) * 100)
        }
    except Exception as e:
        st.error(f"Erreur analyse audio : {e}")
        return {}

def analyze_with_llm(video_features, model_choice):
    df_text = "\n".join([f"- {k}: {v}" for k, v in video_features.items()])
    prompt = f"""
Tu es un expert en marketing vidéo. Analyse cette vidéo à partir de ses caractéristiques :
{df_text}

Attribue un score (0-100) pour chaque aspect (description, audio, visuel, changements de plans, objets détectés).
Calcule ensuite un **score final moyen**.
Donne enfin **trois recommandations** pour améliorer chaque aspect.
"""

    # Configuration pour Gemini et OpenAI
    try:
        if model_choice == "Gemini":
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("models/gemini-flash-latest")
            response = model.generate_content(prompt)
            text_output = response.text.strip()

        elif model_choice in ["ChatGPT", "GPT"]:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            text_output = response.choices[0].message.content.strip()

        else:
            raise ValueError(f"Modèle non reconnu : {model_choice}")

        # ✅ Affichage Streamlit Markdown (rendu riche)
        return st.markdown(text_output)

    except Exception as e:
        st.error(f"Erreur lors de l'analyse avec {model_choice} : {e}")
        return ""
# ============================
# INTERFACE STREAMLIT
# ============================

st.title("🎬 Analyse Marketing Vidéo")
st.markdown("#### 👨‍💻 Auteurs : GBODOGBE Zinsou René")
st.markdown("---")

st.header("1️⃣ Importation de la vidéo")

option = st.radio("Choisissez la source :", ["Lien YouTube", "Téléverser depuis l’ordinateur"])

video_path, metadata = None, {}

if option == "Lien YouTube":
    url = st.text_input("📥 URL YouTube :")
    if st.button("Télécharger la vidéo"):
        with st.spinner("Téléchargement en cours..."):
            video_path, metadata = download_video(url)
        if video_path:
            st.success("✅ Téléchargée avec succès !")
            st.video(video_path)
            st.session_state.video_path = video_path
            st.session_state.metadata = metadata

else:
    uploaded = st.file_uploader("📤 Téléversez une vidéo :", type=["mp4", "webm", "mkv"])
    if uploaded and st.button("Enregistrer et convertir"):
        with st.spinner("Conversion..."):
            video_path, metadata = save_uploaded_video(uploaded)
        if video_path:
            st.video(video_path)
            st.session_state.video_path = video_path
            st.session_state.metadata = metadata

# ============================
# 2️⃣ EXTRACTION DES CARACTÉRISTIQUES
# ============================

st.markdown("---")
st.header("2️⃣ Extraction des caractéristiques")

if st.button("Extraire les caractéristiques"):
    if "video_path" in st.session_state:
        vp = st.session_state.video_path
        md = st.session_state.metadata
        st.info("Analyse en cours...")

        try:
            tech = get_video_metadata(vp)
            vis = analyze_visual_quality(vp)
            aud = analyze_audio(vp)
            obj = analyze_video_with_key(vp)

            all_feat = {"title": md.get("title"), "description": md.get("description"), **tech, **vis, **aud, **obj}
            st.dataframe(pd.DataFrame([all_feat]))
            st.session_state.features = all_feat

        except Exception as e:
            st.error(f"Erreur d’analyse : {e}")
    else:
        st.warning("⚠️ Importez une vidéo d’abord.")

# ============================
# 3️⃣ ANALYSE AVEC LLM
# ============================

st.markdown("---")
st.header("3️⃣ Analyse avec les LLMs")

model_choice = st.selectbox("Choisissez le modèle :", ["ChatGPT", "Gemini"])

if st.button("Analyser avec le modèle choisi"):
    if "features" in st.session_state:
        st.info("Analyse LLM en cours...")
        report = analyze_with_llm(st.session_state.features, model_choice)
    else:
        st.warning("⚠️ Extraire d’abord les caractéristiques.")
