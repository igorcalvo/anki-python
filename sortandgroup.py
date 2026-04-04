from wordfreq import zipf_frequency
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

nltk.data.path.append('/home/calvo/nltk_data')

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "./clean.txt"
OUTPUT_FILE = "./processed.txt"

TARGET_SIZE = 1000
NUM_TOPICS = 6

# -----------------------------
# HELPERS
# -----------------------------
def extract_word(line):
    return line.split("|", 1)[0].strip().lower()


def difficulty(word):
    return zipf_frequency(word, "en")


def same_root(a, b):
    return (
        a[:4] == b[:4] or
        a[:3] == b[:3]
    )


# -----------------------------
# WORDNET ANTONYMS (REAL)
# -----------------------------
def get_antonyms(word):
    antonyms = set()

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().lower().replace("_", "-"))

    return antonyms


# -----------------------------
# SEMANTIC ORDER
# -----------------------------
def semantic_order(cluster_items, all_words, embeddings):
    if not cluster_items:
        return []

    word_to_index = {w: i for i, w in enumerate(all_words)}

    remaining = cluster_items[:]

    # start from easiest
    remaining.sort(reverse=True, key=lambda x: x[0])
    current = remaining.pop(0)

    ordered = [current]

    while remaining:
        current_word = current[1]
        current_vec = embeddings[word_to_index[current_word]]
        found = False

        # 🔥 P0: smarter anchor (semantic + easy)
        if len(ordered) % 5 == 0:
            best_idx = None
            best_score = -1

            for i, (score_val, w, _) in enumerate(remaining):
                vec = embeddings[word_to_index[w]]

                cosine = np.dot(current_vec, vec) / (
                    np.linalg.norm(current_vec) * np.linalg.norm(vec)
                )

                score = 0.7 * cosine + 0.3 * (score_val / 7)

                if score > best_score:
                    best_score = score
                    best_idx = i

            current = remaining.pop(best_idx)
            ordered.append(current)
            continue

        # 🔥 P1: antonyms
        if current_word in ANTONYM_MAP:
            for i, (_, w, _) in enumerate(remaining):
                if w in ANTONYM_MAP[current_word]:
                    current = remaining.pop(i)
                    ordered.append(current)
                    found = True
                    break

        if found:
            continue

        # 🔥 P2: same root
        for i, (_, w, _) in enumerate(remaining):
            if same_root(current_word, w):
                current = remaining.pop(i)
                ordered.append(current)
                found = True
                break

        if found:
            continue

        # 🔥 P3: semantic similarity + smoothness
        best_idx = None
        best_score = -1

        for i, (score_val, w, _) in enumerate(remaining):
            vec = embeddings[word_to_index[w]]

            cosine = np.dot(current_vec, vec) / (
                np.linalg.norm(current_vec) * np.linalg.norm(vec)
            )

            difficulty_score = score_val / 7
            smoothness = 1 - abs(difficulty_score - (current[0] / 7))

            score = (
                0.55 * cosine +
                0.25 * smoothness +
                0.15 * difficulty_score -
                0.05 * abs(len(w) - len(current_word))
            )

            if score > best_score:
                best_score = score
                best_idx = i

        current = remaining.pop(best_idx)
        ordered.append(current)

    return ordered


# -----------------------------
# LOAD
# -----------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if "|" in line]


# -----------------------------
# FILTER + SCORE
# -----------------------------
scored = []

for line in lines:
    word = extract_word(line)
    score = difficulty(word)

    if score < 2.5:
        continue

    scored.append((score, word, line))


scored.sort(reverse=True, key=lambda x: x[0])
scored = scored[:TARGET_SIZE]

print(f"📊 Selected {len(scored)} words")


# -----------------------------
# EMBEDDINGS
# -----------------------------
print("🧠 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

words = [w for _, w, _ in scored]

print("🧠 Encoding words...")
embeddings = model.encode(words, show_progress_bar=True)


# -----------------------------
# BUILD ANTONYM MAP (SYMMETRIC)
# -----------------------------
print("⚔️ Building antonym map...")

ANTONYM_MAP = {}

for word in words:
    ants = get_antonyms(word)
    ants = [a for a in ants if a in words]

    for ant in ants:
        ANTONYM_MAP.setdefault(word, set()).add(ant)
        ANTONYM_MAP.setdefault(ant, set()).add(word)

print(f"⚔️ Found antonyms for {len(ANTONYM_MAP)} words")


# -----------------------------
# CLUSTERING
# -----------------------------
print("📦 Clustering...")
kmeans = KMeans(n_clusters=NUM_TOPICS, random_state=42, n_init="auto")
labels = kmeans.fit_predict(embeddings)


# -----------------------------
# GROUP BY TOPIC
# -----------------------------
clusters = {}

for (score, word, line), label in zip(scored, labels):
    clusters.setdefault(label, []).append((score, word, line))


# -----------------------------
# ORDER INSIDE CLUSTERS
# -----------------------------
print("🔗 Ordering words inside each topic...")

for label in clusters:
    clusters[label] = semantic_order(clusters[label], words, embeddings)


# -----------------------------
# ORDER TOPICS
# -----------------------------
def avg_score(cluster):
    return sum(s for s, _, _ in cluster) / len(cluster)

ordered_labels = sorted(
    clusters.keys(),
    key=lambda l: avg_score(clusters[l]),
    reverse=True
)


# -----------------------------
# 🔥 GLOBAL ANTONYM FIX
# -----------------------------
print("🔀 Applying global antonym pairing...")

final_lines = []
used = set()

word_to_line = {word: line for _, word, line in scored}

for label in ordered_labels:
    for _, word, line in clusters[label]:
        if word in used:
            continue

        final_lines.append(line)
        used.add(word)

        if word in ANTONYM_MAP:
            for ant in ANTONYM_MAP[word]:
                if ant not in used:
                    final_lines.append(word_to_line[ant])
                    used.add(ant)


# -----------------------------
# WRITE OUTPUT
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in final_lines:
        f.write(line + "\n")


print(f"\n🎉 Done! {len(final_lines)} words grouped into {NUM_TOPICS} topics")
print(f"📁 Output: {OUTPUT_FILE}")
