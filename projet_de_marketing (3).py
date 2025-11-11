# Installation des bibliothèques
# %pip install yt-dlp google-generativeai google-cloud-videointelligence

#Importation des bibliothèques

import subprocess
import json
import os
import cv2
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import tempfile
import shutil
import os
import io
import google.generativeai as genai
from google.colab import userdata
from IPython.display import Markdown, display
from google.oauth2 import service_account
from google.cloud import videointelligence

"""**Télechargement de la vidéo youtube**"""

# ===========================
#  Insérer le lien de la vidéo Youtube dont vous souhaitez télécharger
# ===========================
video_urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Astley (already there, usually works)
]

# ===========================
#  Dossier de téléchargement
# ===========================
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)


# ===========================
#  Télécharger  les métadonnées de la vidéo
# ===========================
video_metadata = []

for url in tqdm(video_urls, desc="Téléchargement des métadonnées YouTube"):
    try:
        # Récupère les métadonnées au format JSON
        result = subprocess.run(['yt-dlp', '-j', url], capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        video_metadata.append(metadata)

    except subprocess.CalledProcessError as e:
        print(f" Erreur lors du téléchargement de métadonnées pour {url}: {e}")
        # Print the standard error for more details
        print(f"   Stderr: {e.stderr}")
    except json.JSONDecodeError as e:
        print(f"Erreur JSON pour {url}: {e}")

print(f"\n Métadonnées téléchargées pour {len(video_metadata)} vidéos.")


# ===========================
#  Télécharger les vidéos
# ===========================
for video in tqdm(video_metadata, desc="Téléchargement des vidéos"):
    try:
        # Utilise yt-dlp pour télécharger la meilleure qualité vidéo et audio séparément, puis les muxer
        video_title = video.get('title', 'video').replace('/', '_').replace('\\', '_')
        output_template = os.path.join(VIDEO_DIR, f"{video_title}.%(ext)s")

        subprocess.run([
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]', # Prefers mp4 if available
            '-o', output_template,
            video.get('webpage_url')
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f" Erreur lors du téléchargement de la vidéo pour {video.get('webpage_url')}: {e}")
        print(f"   Stderr: {e.stderr}")

print("\n Téléchargement des vidéos terminé.")

print(f"The videos are downloaded to the following directory: {VIDEO_DIR}")

"""**Extraction des caracteristqiues des vidéos**

Dans un premier temps nous allons extraire quelques méta-données (title, description) de la vidéo et calculer enfin des métriques sur le visuel et l'audio de la vidéo

**Extraction des méta-données**
"""

video_features = []

for video in video_metadata:
    features = {
        'video_id': video.get('id'),
        'title': video.get('title'),
        'channel_name': video.get('channel'),
        'view_count': video.get('view_count'),
        'like_count': video.get('like_count'),
        'comment_count': video.get('comment_count'),
        'duration': video.get('duration'),
        'transcript': video.get('requested_downloads', [{}])[0].get('requested_formats', [{}])[0].get('protocol') if video.get('requested_downloads') else None
    }
    video_features.append(features)

print(f"Extracted features for {len(video_features)} videos.")
print(video_features)

"""**Calcul des métriques sur l'audio et sur le visuel.**

Les métriques sur sur l'audio et le visuel de la vidéo seront calculées en utilisant les deux fonctions ci-dessus.
"""

# ===========================
#  Fonction pour l'extraction de quelques métadonnées locales de la vidéo: durée, width, height, video_codec_audio_codec
# ===========================

def get_video_metadata(video_path):
    """Extrait les métadonnées techniques de la vidéo via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_name,codec_type,width,height",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    duration = float(data["format"].get("duration", 0))
    video_codec, audio_codec, width, height = None, None, None, None

    for stream in data["streams"]:
        if stream["codec_type"] == "video":
            video_codec = stream.get("codec_name")
            width = stream.get("width")
            height = stream.get("height")
        elif stream["codec_type"] == "audio":
            audio_codec = stream.get("codec_name")

    return {
        #"duration_sec": duration,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "width": width,
        "height": height
    }

# ===========================
# Fonction pour l'extraction de quelques métriques sur la qualité du visuel en prenant un échantillon de 30 images.
# ===========================
def analyze_visual_quality(video_path, sample_frames=30):
    """Analyse la qualité visuelle (luminosité, contraste, netteté, couleur dominante)."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, frame_count - 1, sample_frames).astype(int)

    brightness, contrast, sharpness, colors = [], [], [], []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(gray.mean())
        contrast.append(gray.std())
        sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        resized = cv2.resize(frame, (100, 100))
        data = resized.reshape(-1, 3)
        dominant_color = tuple(np.round(np.mean(data, axis=0)).astype(int))
        colors.append(dominant_color)

    cap.release()

    # Ensure colors is a NumPy array before calculating mean
    colors_np = np.array(colors)

    mean_color = np.mean(colors_np, axis=0) if colors_np.size > 0 else np.array([0, 0, 0])
    mean_color_hex = '#%02x%02x%02x' % tuple(mean_color.astype(int))

    return {
        "mean_brightness": np.mean(brightness) if brightness else None,
        "mean_contrast": np.mean(contrast) if contrast else None,
        "mean_sharpness": np.mean(sharpness) if sharpness else None,
        "dominant_color_rgb": tuple(mean_color.astype(int)) if colors_np.size > 0 else None,
        "dominant_color_hex": mean_color_hex if colors_np.size > 0 else None
    }


# ===========================
#  Fonction pour l'extraction de quelques métriques sur l'audio.
# ===========================
def analyze_audio(video_path):
    """Analyse la piste audio : volume, bruit et spectre."""
    audio_path = "temp_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        y, sr = librosa.load(audio_path, sr=None)
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        noise_level = np.mean(np.abs(y[y < 0.01])) * 100
        os.remove(audio_path)
        return {
            "rms_volume": float(rms),
            "spectral_centroid": float(spectral_centroid),
            "noise_level": float(noise_level)
        }
    except Exception as e:
        print(f"Erreur d’analyse audio : {e}")
        return {"rms_volume": None, "spectral_centroid": None, "noise_level": None}

# ===========================
#  Combiner toutes les caractéristiques de la vidéo à utiliser
# ===========================
video_features = []

# Set the absolute path for the video directory
VIDEO_DIR = "/content/videos"

for video in tqdm(video_metadata, desc="Analyse complète des vidéos"):
    # Métadonnées YouTube
    yt_data = {
        'video_id': video.get('id'),
        'title': video.get('title'),
        'description': video.get('description'),
        'duration': video.get('duration')
    }

    # Construct the expected local video path
    video_title = video.get('title', 'video').replace('/', '_').replace('\\', '_') # Sanitize title for filename
    # Try common extensions
    video_path = None
    for ext in ['mp4', 'webm', 'mkv']:
        potential_path = os.path.join(VIDEO_DIR, f"{video_title}.{ext}")
        if os.path.exists(potential_path):
            video_path = potential_path
            break

    if video_path and os.path.exists(video_path):
        processed_video_path = video_path
        temp_dir = None # Initialize temp_dir outside try for finally block
        try:
            technical_features = get_video_metadata(video_path)

            # Check if video codec is AV1 or resolution is too high for direct OpenCV processing
            needs_processing_for_cv2 = False
            if technical_features.get('video_codec') == 'av1' or technical_features.get('height', 0) > 1080:
                needs_processing_for_cv2 = True

            if needs_processing_for_cv2:
                print(f"  > Processing {video.get('id')} for OpenCV compatibility (downscaling/re-encoding)...")
                temp_dir = tempfile.mkdtemp()
                processed_video_path = os.path.join(temp_dir, f"temp_{video.get('id')}.mp4")

                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", "scale=1280:-1",  # Scale to 720p width, maintain aspect ratio
                    "-c:v", "libx264", "-preset", "fast", "-crf", "28", # Use H.264 for compatibility
                    "-an", # No audio stream in the temp video for visual analysis
                    processed_video_path
                ]

                # Run ffmpeg with error handling
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    print(f"  FFmpeg Error for {video.get('id')}:")
                    print(f"    STDOUT: {result.stdout}")
                    print(f"    STDERR: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, ffmpeg_cmd, output=result.stdout, stderr=result.stderr)

                print(f"  > Temporary video created: {processed_video_path}")

            # Analyze visual and audio features using the appropriate paths
            visual_features = analyze_visual_quality(processed_video_path)
            audio_features = analyze_audio(video_path) # analyze_audio still uses original video path

            combined_features = {**yt_data, **technical_features, **visual_features, **audio_features}
            video_features.append(combined_features)

        except subprocess.CalledProcessError as e:
            print(f"Erreur lors du traitement FFmpeg ou autre sous-processus pour {video.get('id')}: {e}")
            video_features.append(yt_data)
        except Exception as e:
            print(f"Erreur lors de l'analyse locale de {video.get('id')}: {e}")
            video_features.append(yt_data)
        finally:
            # Clean up temporary directory if created
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"  > Cleaned up temporary directory: {temp_dir}")

    else:
        print(f"Vidéo non trouvée localement pour l'ID {video.get('id')}. Ajout des métadonnées YouTube seulement.")
        video_features.append(yt_data)


print(f"\n Caractéristiques collectées pour {len(video_features)} vidéos.")

"""**Extraction des caractéristiques de la vidéo en utilisant google intelligence video.**


 Ce package va nous permettre de compter le nombre d'objets present dans la vidéo ainsi que le nombre de changement de plan pu de scène.
"""

# Exemple : KEY_PATH = "/content/my-service-account-key.json"
KEY_PATH = "/content/civil-oarlock-476523-r4-4a3bf0be8497.json" # <--- METTEZ VOTRE VRAI CHEMIN ICI
VIDEO_PATH = "/content/videos/Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp4"                # adapte au nom du fichier uploadé

def analyze_video_with_key(video_path, key_path=KEY_PATH, save_json="/content/video_analysis_results.json"):
    """
    Analyse une vidéo locale ou GCS avec la Video Intelligence API.
    - video_path : '/content/ma_video.mp4' ou 'gs://bucket/video.mp4'
    - key_path : chemin vers la clé JSON
    - save_json : emplacement pour sauvegarder les résultats
    """
    # Création des credentials et du client (sécurisé)
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)

    features = [
        videointelligence.Feature.SHOT_CHANGE_DETECTION,
        videointelligence.Feature.OBJECT_TRACKING
    ]

    if video_path.startswith("gs://"):
        request = {"input_uri": video_path, "features": features}
        print(f"Analyse d'une vidéo GCS : {video_path}")
    else:
        with io.open(video_path, "rb") as f:
            input_content = f.read()
        request = {"input_content": input_content, "features": features}
        print(f"Analyse d'une vidéo locale : {video_path}")

    # Lancer l'annotation
    operation = client.annotate_video(request=request)
    print("⏳ Analyse en cours (cela peut prendre plusieurs minutes)...")
    result = operation.result(timeout=900)  # timeout en secondes

    # Récupérer les résultats et les formater
    annotation_result = result.annotation_results[0]
    out = {}

    # Shot changes
    out["shot_changes_count"] = len(annotation_result.shot_annotations) if annotation_result.shot_annotations else 0

    # Object tracking (exemples)
    objects = []
    for obj in (annotation_result.object_annotations or [])[:50]:  # limite pour éviter trop de sorties
        desc = obj.entity.description
        conf = getattr(obj, "confidence", None)
        seg_start = getattr(obj.segment.start_time_offset, "seconds", 0) + getattr(obj.segment.start_time_offset, "microseconds", 0)/1e6
        seg_end = getattr(obj.segment.end_time_offset, "seconds", 0) + getattr(obj.segment.end_time_offset, "microseconds", 0)/1e6
        objects.append({"description": desc, "confidence": conf, "start_s": seg_start, "end_s": seg_end})
    out["object_tracking"] = objects

    # Sauvegarde JSON
    with open(save_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Analyse terminée. Résultats sauvegardés dans {save_json}")

    return out

# Exécution (adapte VIDEO_PATH au nom réel)
results = analyze_video_with_key(VIDEO_PATH)
print("Résumé :", {k: (len(v) if isinstance(v, list) else v) for k,v in results.items()})

"""## Create dataframe"""

# ===========================
#  Créer le DataFrame final et ajouter les résultats de Video Intelligence
# ===========================
if video_features and results:
    # Extract labels descriptions
    video_features[0]['shot_changes_count'] = results.get('shot_changes_count')
    video_features[0]['object_tracking_count'] = len(results.get('object_tracking', []))

df = pd.DataFrame(video_features)
print("\nCaractéristiques extraites :\n")
display(df.head())

"""## Analyse des vidéos avec les LLMs

Il s'agit d'analyse nos vidéos par les LLMs . Le LLM nous fournira un score d'anlyse compris en 0 et 100 de chacune des vidéos.
"""

# ===========================================================
#  Analyse de la qualité vidéo avec Gemini à partir du DataFrame
# ===========================================================

# Configuration de l'API
GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(" Clé API manquante ! Ajoute-la dans Colab > User secrets.")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialisation du modèle
model = genai.GenerativeModel("models/gemini-flash-latest")

# Prompt général pour l’analyse
base_prompt = """
Tu es un expert en marketing vidéo. Analyse cette vidéo à partir de ses caractéristiques :
{video_features}

Attribue un score (0-100) pour chaque aspect (description, audio, visuel, changements de plans, objets détectés).
Calcule ensuite un **score final moyen**.
Donne enfin **trois recommandations** pour améliorer chaque aspect.

"""

# Analyse des vidéos du DataFrame
for i, row in df.iterrows():
    # Transformation automatique des caractéristiques de la ligne en texte
    video_features = "\n".join([f"- {col}: {row[col]}" for col in df.columns])

    # Création du prompt complet
    user_prompt = base_prompt.format(video_features=video_features)

    print(f"\n{'='*80}")
    print(f"Vidéo {i+1}: {row.get('title', 'Sans titre')}")
    print(f"{'='*80}\n")

    try:
        response = model.generate_content(user_prompt)
        output_text = response.text or ""
    except Exception as e:
        output_text = f"Erreur lors de l'analyse de la vidéo {i+1}: {e}"

    # --- Affichage brut et lisible

display(Markdown(output_text))

"""## Analyse avec chatgpt"""

# ===========================================================
#  Analyse de la qualité vidéo avec ChatGPT à partir du DataFrame
# ===========================================================

from openai import OpenAI
from google.colab import userdata
from IPython.display import display, Markdown

# Configuration de l'API OpenAI
OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Clé API OpenAI manquante ! Ajoute-la dans Colab > User secrets.")
client = OpenAI(api_key=OPENAI_API_KEY)

#  Choix du modèle
MODEL_NAME = "gpt-4o-mini"

#  Prompt général pour l’analyse
base_prompt = """
Tu es un expert en marketing vidéo. Analyse cette vidéo à partir de ses caractéristiques :
{video_features}

Attribue un score (0-100) pour chaque aspect (description, audio, visuel, changements de plans, objets détectés).
Calcule ensuite un **score final moyen**.
Donne enfin **trois recommandations** pour améliorer chaque aspect.

"""
# Boucle d’analyse des vidéos du DataFrame
for i, row in df.iterrows():
    # Génération du bloc de caractéristiques
    video_features = "\n".join([f"- {col}: {row[col]}" for col in df.columns])

    # Création du prompt complet
    user_prompt = base_prompt.format(video_features=video_features)

    print(f"\n{'='*80}")
    print(f"Vidéo {i+1}: {row.get('title', 'Sans titre')}")
    print(f"{'='*80}\n")

    try:
        # Appel du modèle ChatGPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Tu es un assistant expert en analyse marketing vidéo."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        # Extraction du texte brut du modèle
        output_text = response.choices[0].message.content
    except Exception as e:
        output_text = f" Erreur lors de l'analyse de la vidéo {i+1}: {e}"

    # --- Affichage brut du texte généré
    display(Markdown(output_text))