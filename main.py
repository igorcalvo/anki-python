import genanki
import re
import requests
from pathlib import Path
from gtts import gTTS
from deep_translator import GoogleTranslator

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "clean.txt"
OUTPUT_DECK = "english_vocab.apkg"

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

    try:
        print(f"   🖼️ Fetching image: {word}")

        url = "https://api.pexels.com/v1/search"

        headers = {
            "Authorization": PEXELS_API_KEY
        }

        params = {
            "query": word,
            "per_page": 1
        }

        r = requests.get(url, headers=headers, params=params, timeout=10)

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

# -----------------------------
# LOAD DATA
# -----------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

words_data = []

for line in lines:
    try:
        word, word_type, sentences = line.strip().split("|")

        sentence_list = [s.strip() for s in sentences.split(";") if s.strip()]

        words_data.append({
            "word": word,
            "type": word_type,
            "sentences": sentence_list[:3]
        })

    except:
        continue


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
            "afmt" : """
            <div class="answer">
            
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
    .card {
        font-family: Arial;
    }
    
    .center {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100vh;
        text-align: center;
    }
    
    .type {
        color: gray;
        font-size: 18px;
    }
    
    /* ANSWER SIDE */
    .answer {
        text-align: left;
        padding: 10px;
    }
    
    .translation {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .examples {
        font-size: 18px;
        line-height: 1.6;
    }
    
    .highlight {
        color: #ff5555;
        font-weight: bold;
    }
    """
)

deck = genanki.Deck(1234567891, "English Deck")

media_files = []

# -----------------------------
# IMAGE SETUP
# -----------------------------
image_filename = MEDIA_DIR / "shared.jpg"

if not image_filename.exists():
    with open(STATIC_IMAGE_PATH, "rb") as src, open(image_filename, "wb") as dst:
        dst.write(src.read())

media_files.append(str(image_filename))
image_field = f'<img src="{image_filename.name}">'

# -----------------------------
# CREATE NOTES
# -----------------------------
for idx, item in enumerate(words_data[:20], 1):
    word = item["word"]
    word_type = item["type"]
    sentences = item["sentences"]

    print(f"\n=== [{idx}] Processing: {word} ===")

    # 🔊 audio
    audio_file = make_audio(word)
    audio_field = f"[sound:{audio_file}]" if audio_file else ""

    if audio_file:
        media_files.append(str(MEDIA_DIR / audio_file))

    # 🌍 translation (word)
    translation = translate(word)

    # 💬 examples
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

    deck.add_note(note)

    print(f"✅ Done: {word}")

# -----------------------------
# EXPORT
# -----------------------------
package = genanki.Package(deck)
package.media_files = media_files
package.write_to_file(OUTPUT_DECK)

print("\n🎉 Deck generated:", OUTPUT_DECK)
