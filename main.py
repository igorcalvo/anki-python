import genanki
import re
import requests
import time
from pathlib import Path
from gtts import gTTS
from deep_translator import GoogleTranslator

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "processed3000.txt"
OUTPUT_DECK = "en_pt_3000_beginner.apkg"
DECK_ID = 1876543210
DECK_NAME = "English-Portugues ~3000 Common Words/Palarvras Comuns"
LIMIT = 3200

MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

translator = GoogleTranslator(source="en", target="pt")

# -----------------------------
# HELPERS
# -----------------------------
def translate(text):
    try:
        print(f"   🌍 Translating: {text[:40]}...")
        return translator.translate(text[:200])
    except Exception as e:
        print("   ❌ Translation error:", e)
        return text


def make_audio(word):
    safe = re.sub(r"[^a-zA-Z0-9]", "_", word)
    filename = MEDIA_DIR / f"{safe}.mp3"

    if not filename.exists():
        try:
            print(f"   🔊 Generating audio: {word}")
            tts = gTTS(word)
            tts.save(filename)
        except Exception as e:
            print("   ❌ TTS error:", e)
            return ""

    return filename.name


def highlight_word(sentence, word):
    return re.sub(
        rf"\b({re.escape(word)})\b",
        r'<span class="highlight">\1</span>',
        sentence,
        flags=re.IGNORECASE
    )


def load_pexels_key():
    with open("./pexelskey", "r") as f:
        return f.read().strip()


PEXELS_API_KEY = load_pexels_key()

def get_image(word, filename):
    if filename.exists():
        return True

    for attempt in range(3):  # retry
        try:
            print(f"   🖼️ Fetching image: {word}")

            url = "https://api.pexels.com/v1/search"

            headers = {"Authorization": PEXELS_API_KEY}
            params = {"query": word, "per_page": 1}

            r = requests.get(url, headers=headers, params=params, timeout=10)

            if r.status_code == 429:
                print("   ⏳ Rate limited, retrying...")
                time.sleep(10 * (attempt + 1))
                continue

            if r.status_code != 200:
                print("   ❌ Pexels API error:", r.status_code)
                return False

            data = r.json()

            if not data.get("photos"):
                print("   ⚠️ No image found")
                return False

            img_url = data["photos"][0]["src"]["medium"]
            img_data = requests.get(img_url, timeout=10).content

            with open(filename, "wb") as f:
                f.write(img_data)

            return True

        except Exception as e:
            print("   ❌ Image error:", e)

    return False

def prepare_image(word, media_files):
    safe_word = re.sub(r"[^a-zA-Z0-9]", "_", word)
    image_filename = MEDIA_DIR / f"{safe_word}.jpg"

    if get_image(word, image_filename):
        if str(image_filename) not in media_files:
            media_files.append(str(image_filename))
        return f'<img src="{image_filename.name}">'

    fallback = MEDIA_DIR / "fallback.jpg"

    if not fallback.exists():
        try:
            with open("./images/testimage.jpg", "rb") as src, open(fallback, "wb") as dst:
                dst.write(src.read())
        except Exception as e:
            print("   ❌ Fallback image error:", e)
            return ""

    if str(fallback) not in media_files:
        media_files.append(str(fallback))

    return f'<img src="{fallback.name}">'


# -----------------------------
# LOAD DATA
# -----------------------------
def load_words():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []

    for line in lines:
        try:
            word, word_type, sentences = line.strip().split("|")

            sentence_list = [s.strip() for s in sentences.split(";") if s.strip()]

            data.append({
                "word": word,
                "type": word_type,
                "sentences": sentence_list[:3]
            })

        except:
            continue

    return data

words_data = load_words()

# -----------------------------
# MODEL
# -----------------------------
model = genanki.Model(
    1234567890,
    "Clean English Model",
    fields=[
        {"name": "Word"},
        {"name": "Type"},
        {"name": "Audio"},
        {"name": "Translation"},
        {"name": "Examples"},
        {"name": "Image"},
    ],
    templates=[
        {
            "name": "Card",
            "qfmt": """
                <div class="center">
                    <h1>{{Word}}</h1>
                    <div class="type">({{Type}})</div>
                    <br>
                    {{Audio}}
                </div>
            """,
            "afmt": """
            <div class="answer">
            
                <div class="answer-header">
                    <strong>{{Word}}</strong> ({{Type}})
                    <br>
                    {{Audio}}
                </div>
            
                <hr>
            
                <div class="translation">
                    {{Translation}}
                </div>
            
                <div class="examples">
                    {{Examples}}
                </div>
            
                <br>
                {{Image}}
            
            </div>
            """,
        }
    ],
    css="""
    .card { font-family: Arial; }
    .center {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100vh;
        text-align: center;
    }
    .type { color: gray; font-size: 18px; }
    .answer { text-align: left; padding: 10px; }
    .translation {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .examples { font-size: 18px; line-height: 1.6; }
    .highlight { color: #ff5555; font-weight: bold; }
    .answer-header {
        text-align: center;
        font-size: 20px;
        margin-bottom: 10px;
    }
    .answer-header strong { font-size: 24px; }
    """
)

# -----------------------------
# CORE LOGIC (REUSABLE)
# -----------------------------
def process_word(item, idx):
    word = item["word"]
    word_type = item["type"]
    sentences = item["sentences"]

    print(f"\n=== [{idx}] Processing: {word} ===")

    local_media = []

    image_field = prepare_image(word, local_media)

    audio_file = make_audio(word)
    audio_field = f"[sound:{audio_file}]" if audio_file else ""

    if audio_file:
        local_media.append(str(MEDIA_DIR / audio_file))

    translation = translate(word)

    example_html = ""
    for s in sentences:
        highlighted = highlight_word(s, word)
        pt = translate(s)

        example_html += f"""
        <div>
            {highlighted}<br>
            <i>{pt}</i>
        </div>
        <br>
        """

    note = genanki.Note(
        model=model,
        fields=[
            word,
            word_type,
            audio_field,
            translation,
            example_html,
            image_field,
        ],
    )

    print(f"✅ Done: {word}")

    return note, local_media


# -----------------------------
# SEQUENTIAL RUNNER
# -----------------------------
def run_sequential(limit=LIMIT):
    deck = genanki.Deck(DECK_ID, DECK_NAME)
    media_files = []

    for idx, item in enumerate(words_data[:limit], 1):
        note, local_media = process_word(item, idx)

        deck.add_note(note)

        for m in local_media:
            if m not in media_files:
                media_files.append(m)

    package = genanki.Package(deck)
    package.media_files = media_files
    package.write_to_file(OUTPUT_DECK)

    print("\n🎉 Deck generated:", OUTPUT_DECK)


# -----------------------------
# ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    print("Running sequential mode (debug)...")
    run_sequential(LIMIT)
