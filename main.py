import genanki
import requests
import re
from pathlib import Path
from deep_translator import GoogleTranslator

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_DECK = "english_vocab.apkg"
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

NUM_WORDS = 5
STATIC_IMAGE_PATH = Path("./images/testimage.jpg")

translator = GoogleTranslator(source="en", target="pt")

# -----------------------------
# HELPERS
# -----------------------------
def download_file(url, filename):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(filename, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print("Download failed:", e)
    return False


def translate_word(word):
    try:
        return translator.translate(word)
    except Exception as e:
        print("Translation error:", e)
        return f"[PT] {word}"


# -----------------------------
# PARSE FILE (SIMPLIFIED)
# -----------------------------
words_data = []

with open("./content.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # match: "word (type)"
        match = re.match(r"^([a-zA-Z\-]+)\s*\(([^)]+)\)", line)

        if match:
            word = match.group(1)
            word_type = match.group(2)

            words_data.append({
                "word": word,
                "type": word_type
            })

        if len(words_data) >= NUM_WORDS:
            break


# -----------------------------
# MODEL
# -----------------------------
model = genanki.Model(
    1234567890,
    "Simple English Model",
    fields=[
        {"name": "Word"},
        {"name": "Type"},
        {"name": "Translation"},
        {"name": "Image"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": """
                <h2>{{Word}}</h2>
                <i>({{Type}})</i>
            """,
            "afmt": """
                {{FrontSide}}
                <hr id="answer">

                <b>Translation:</b><br>
                {{Translation}}<br><br>

                {{Image}}
            """,
        }
    ],
)

deck = genanki.Deck(1234567891, "English Vocab Deck")

media_files = []

# -----------------------------
# STATIC IMAGE
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
for item in words_data:
    word = item["word"]
    word_type = item["type"]

    translation = translate_word(word)

    note = genanki.Note(
        model=model,
        fields=[
            word,
            word_type,
            translation,
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
