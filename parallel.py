import genanki
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🔥 import EVERYTHING from main
from main import (
    words_data,
    process_word,
)

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_DECK = "en_pt_1000_beginner.apkg"
MAX_WORKERS = 6

# 🔥 generate once and KEEP THIS VALUE
# random.randrange(1 << 30, 1 << 31)
DECK_ID = 1876543210

deck = genanki.Deck(DECK_ID, "English-Portugues ~1000 Beginner Common Words/Iniciante Palarvras Comuns")
media_files = []

# -----------------------------
# PARALLEL EXECUTION
# -----------------------------
results = [None] * len(words_data)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {}

    for idx, item in enumerate(words_data, 1):
        future = executor.submit(process_word, item, idx)
        futures[future] = idx - 1

    completed = 0

    for future in as_completed(futures):
        index = futures[future]

        try:
            results[index] = future.result()
        except Exception as e:
            print(f"❌ Error at index {index}: {e}")
            results[index] = None

        completed += 1
        print(f"⚡ Progress: {completed}/{len(words_data)}")

# -----------------------------
# BUILD DECK (ORDER PRESERVED)
# -----------------------------
for i, result in enumerate(results):
    if result is None:
        print(f"⚠️ Skipping failed item at index {i}")
        continue

    note, local_media = result

    deck.add_note(note)

    for m in local_media:
        if m not in media_files:
            media_files.append(m)

# -----------------------------
# EXPORT
# -----------------------------
package = genanki.Package(deck)
package.media_files = media_files
package.write_to_file(OUTPUT_DECK)

print("\n🎉 Deck generated:", OUTPUT_DECK)
