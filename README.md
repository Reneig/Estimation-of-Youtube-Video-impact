# Estimation-of-Youtube-Video-impact
#  Analyse Marketing de Vidéos YouTube avec LLMs et Google Video Intelligence

## Description du projet

Ce projet permet d’**analyser automatiquement des vidéos YouTube** sous un angle **marketing** en combinant afin d'estimer leur impact en terme de score:

- des **caractéristiques techniques et visuelles** (luminosité, contraste, netteté, couleur dominante, codecs, etc.),
- des **métriques audio** (volume, bruit, spectre),
- des **données de la Google Video Intelligence API** (objets détectés, changements de plans),
- et enfin une **évaluation qualitative** par **modèles de langage (LLMs)** tels que **Gemini** et **ChatGPT**.

L’objectif est de produire une **évaluation marketing globale (score 0–100)** accompagnée de **recommandations concrètes** pour améliorer la qualité et l’impact des vidéos.

---

## 📋 Fonctionnalités principales

| Module | Description |
|:-------|:-------------|
| 🎬 **Téléchargement YouTube** | Télécharge automatiquement la vidéo et ses métadonnées à partir de son URL via `yt-dlp`. |
| 🧾 **Extraction de métadonnées** | Récupère le titre, la durée, le nombre de vues, de likes et de commentaires. |
| 📷 **Analyse visuelle** | Calcule la luminosité moyenne, le contraste, la netteté et la couleur dominante. |
| 🎧 **Analyse audio** | Mesure le volume RMS, le bruit et le spectre fréquentiel avec `librosa`. |
| 🤖 **Google Video Intelligence** | Détecte les objets, les scènes et les transitions de plans. |
| 🧩 **Fusion des caractéristiques** | Regroupe toutes les données dans un `DataFrame` exploitable pour l’analyse. |
| 🧠 **Analyse par LLMs (Gemini / ChatGPT)** | Génère des scores et recommandations marketing personnalisées. |

---

## ⚙️ Installation et configuration

### 1️⃣ Installation des dépendances
Pour que le code soit fonctionnel, vous devez installer des packages. Voici le code basique pour l'installation.
%pip install yt-dlp google-generativeai google-cloud-videointelligence librosa opencv-python pandas tqdm

2️⃣ Clés d’API requises

## Google Video Intelligence : 
Pour acceder aux fonctionnalités de google video intelligence vous devez créer une clé de service depuis Google Cloud Console.
Dans le code , télécharger le fichier JSON de votre clé  et indiquer son chemin :
KEY_PATH = "/content/mon-projet-google-key.json"

## Gemini (Google Generative AI) :
Pour acceder aux fonctionnalités de Gemini vous devez créer et stocker votre clé API dans un dossier secret, le nom de l'API sera 
GOOGLE_API_KEY

## OpenAI (ChatGPT) :
Pour acceder aux fonctionnalités de Gemini vous devez créer et stocker votre clé API dans un dossier secret, le nom de l'API sera 
OPENAI_API_KEY

▶️ Utilisation
Étape 1 : Télécharger une vidéo YouTube
video_urls = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]

Le script télécharge automatiquement la vidéo et ses métadonnées dans le dossier /content/videos.
Étape 2 : Extraire les caractéristiques
video_features = analyze_visual_quality(video_path)
audio_features = analyze_audio(video_path)

Étape 3 : Analyser la vidéo avec Google Video Intelligence
results = analyze_video_with_key(VIDEO_PATH, key_path=KEY_PATH)
→ Fournit le nombre de changements de plans et d’objets détectés.

Étape 4 : Fusionner les résultats dans un DataFrame
df = pd.DataFrame(video_features)
display(df.head())

Étape 5 : Analyse par les LLMs
🔹 Avec Gemini

response = model.generate_content(user_prompt)
display(Markdown(response.text))

🔹 Avec ChatGPT

response = client.chat.completions.create(...)
display(Markdown(output_text))

Les deux modèles attribuent des scores par dimension (audio, visuel, description, etc.) et proposent des recommandations marketing.

## Application streamlit

# 🎛️ Application Streamlit — Analyse Marketing Vidéo

## 🧩 Description

Cette application **Streamlit** offre une interface simple et interactive pour exécuter localement le pipeline complet d’**analyse marketing de vidéos YouTube**.  
Elle permet de :
- importer ou téléverser une vidéo,
- extraire automatiquement ses caractéristiques techniques, visuelles et audio,
- utiliser les APIs **Google Video Intelligence** et **LLMs (Gemini / ChatGPT)** pour générer une **analyse marketing détaillée**.

## 🚀 Lancement rapide

### 1️⃣ Installation des dépendances
A la suite des packages installés ci-dessus vous installez le package pour tourner votre application streamlit.
pip install streamlit yt-dlp 

## Quelquetions supplementaires

Mettez dans votre dossier local, le fichier .json de votre clé API de google video intelligence, Créer également un dossier video pour stocker les vidéos téléchargées.
Assurez vous d'assurer d'avoir installer le fichier executablme ffmpeg.

## Lancer l'application.
streamlit run app.py



