import genanki
import re
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

STATIC_IMAGE_PATH = Path("./images/testimage.jpg")

translator = GoogleTranslator(source="en", target="pt")

# -----------------------------
# HELPERS
# -----------------------------
def translate(text):
    try:
        return translator.translate(text[:200])
    except:
        return text


def make_audio(word):
    safe = re.sub(r"[^a-zA-Z0-9]", "_", word)
    filename = MEDIA_DIR / f"{safe}.mp3"

    if not filename.exists():
        try:
            tts = gTTS(word)
            tts.save(filename)
        except Exception as e:
            print("TTS error:", e)
            return ""

    return filename.name


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
            "afmt": """
                {{FrontSide}}
                <hr>

                <b>{{Translation}}</b><br><br>

                {{Examples}}<br><br>

                {{Image}}
            """,
        }
    ],
    css="""
    .card {
        font-family: Arial;
        text-align: center;
    }
    .center {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100vh;
    }
    .type {
        color: gray;
        font-size: 18px;
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
for item in words_data[:20]:  # limit for testing
    word = item["word"]
    word_type = item["type"]
    sentences = item["sentences"]

    # 🔊 audio
    audio_file = make_audio(word)
    audio_field = f"[sound:{audio_file}]" if audio_file else ""

    if audio_file:
        media_files.append(str(MEDIA_DIR / audio_file))

    # 🌍 translation
    translation = translate(f"{word} ({word_type})")

    # 💬 examples
    example_html = ""
    for s in sentences:
        pt = translate(s)
        example_html += f"{s}<br><i>{pt}</i><br><br>"

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


# -----------------------------
# EXPORT
# -----------------------------
package = genanki.Package(deck)
package.media_files = media_files
package.write_to_file(OUTPUT_DECK)

print("Deck generated:", OUTPUT_DECK)
