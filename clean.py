import re

INPUT_FILE = "content.txt"
OUTPUT_FILE = "clean.txt"


def clean_text(text):
    text = re.sub(r"\{\{c\d+::(.*?)\}\}", r"\1", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sentences(html):
    matches = re.findall(r'content-example">(.*?)</li>', html)

    out = []
    for m in matches:
        s = clean_text(m)

        if len(s) > 10 and "see also" not in s.lower():
            out.append(s)

    return out[:3]


# -----------------------------
# LOAD
# -----------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

parts = content.split("\x1f")

clean_lines = []

print("\n--- SANITIZING (FINAL FIX) ---\n")

i = 0
while i < len(parts) - 1:
    word = parts[i].strip()

    # ✅ detect real entry start
    if (
        word
        and "<" not in word
        and not word.startswith("(")
        and i + 1 < len(parts)
        and parts[i + 1].strip().startswith("(")
    ):
        try:
            type_line = parts[i + 1].strip()
            word_type = type_line.strip("()")

            # 🔥 instead of fixed offset, SEARCH forward for examples
            examples_html = ""
            for j in range(i, min(i + 15, len(parts))):
                if "content-example" in parts[j]:
                    examples_html = parts[j]
                    break

            sentences = extract_sentences(examples_html)

            if sentences:
                line = f"{word}|{word_type}|{';'.join(sentences)}"
                clean_lines.append(line)

                print(line)

        except Exception:
            pass

    # ✅ ALWAYS move by 1 (never skip entries)
    i += 1


# -----------------------------
# SAVE
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(clean_lines))

print("\nClean file generated:", OUTPUT_FILE)
