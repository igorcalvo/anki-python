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
NUM_TOPICS = 15  # tweak (8–20 works well)

# -----------------------------
# HELPERS
# -----------------------------
def extract_word(line):
    return line.split("|", 1)[0].strip().lower()


def difficulty(word):
    return zipf_frequency(word, "en")


# 🔥 cache for speed
related_cache = {}

def get_related_words(word):
    if word in related_cache:
        return related_cache[word]

    related = set()

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            related.add(lemma.name().lower().replace("_", "-"))

            if lemma.antonyms():
                related.add(
                    lemma.antonyms()[0].name().lower().replace("_", "-")
                )

    related_cache[word] = related
    return related


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

        # 🔥 PRIORITY 1: synonym / antonym
        related = get_related_words(current_word)

        found = False
        for i, (_, w, _) in enumerate(remaining):
            if w in related:
                current = remaining.pop(i)
                ordered.append(current)
                found = True
                break

        if found:
            continue

        # 🔥 PRIORITY 1: TRUE antonyms (auto-detected)
        if current_word in ANTONYM_MAP:
            targets = ANTONYM_MAP[current_word]
        
            for i, (_, w, _) in enumerate(remaining):
                if w in targets:
                    current = remaining.pop(i)
                    ordered.append(current)
                    found = True
                    break
        
            if found:
                continue
        
        
        # 🔥 PRIORITY 2: WordNet relations
        related = get_related_words(current_word)
        
        for i, (_, w, _) in enumerate(remaining):
            if w in related:
                current = remaining.pop(i)
                ordered.append(current)
                found = True
                break
        
        if found:
            continue

        # 🔥 PRIORITY 3: semantic similarity
        current_vec = embeddings[word_to_index[current_word]]

        best_idx = 0
        best_score = -1

        for i, (score_val, w, _) in enumerate(remaining):
            vec = embeddings[word_to_index[w]]

            cosine = np.dot(current_vec, vec) / (
                np.linalg.norm(current_vec) * np.linalg.norm(vec)
            )

            # hybrid score (semantic + difficulty)
            score = 0.85 * cosine + 0.15 * (score_val / 7)

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


# keep best N
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
# BUILD ANTONYM MAP (AUTO)
# -----------------------------
print("⚔️ Detecting opposites automatically...")

ANTONYM_MAP = {}

SIMILARITY_THRESHOLD = 0.65  # high similarity
MAX_NEIGHBORS = 10

word_to_index = {w: i for i, w in enumerate(words)}

for i, word in enumerate(words):
    vec = embeddings[i]

    similarities = []

    for j, other in enumerate(words):
        if i == j:
            continue

        other_vec = embeddings[j]

        cosine = np.dot(vec, other_vec) / (
            np.linalg.norm(vec) * np.linalg.norm(other_vec)
        )

        if cosine > SIMILARITY_THRESHOLD:
            similarities.append((cosine, other))

    # sort by similarity
    similarities.sort(reverse=True)

    # take top neighbors
    neighbors = [w for _, w in similarities[:MAX_NEIGHBORS]]

    # check WordNet antonyms among neighbors
    antonyms = set()

    wn_related = get_related_words(word)

    for n in neighbors:
        if n in wn_related:
            continue  # skip synonyms

        # 🔥 heuristic: similar but NOT synonym → likely contrast
        antonyms.add(n)

    if antonyms:
        ANTONYM_MAP[word] = antonyms

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
# SEMANTIC ORDER INSIDE EACH TOPIC
# -----------------------------
print("🔗 Ordering words semantically inside each topic...")

for label in clusters:
    clusters[label] = semantic_order(clusters[label], words, embeddings)


# -----------------------------
# ORDER TOPICS (easy → harder)
# -----------------------------
def avg_score(cluster):
    return sum(s for s, _, _ in cluster) / len(cluster)

ordered_labels = sorted(
    clusters.keys(),
    key=lambda l: avg_score(clusters[l]),
    reverse=True
)


# -----------------------------
# WRITE OUTPUT
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for label in ordered_labels:
        for _, _, line in clusters[label]:
            f.write(line + "\n")


print(f"\n🎉 Done! {TARGET_SIZE} words grouped into {NUM_TOPICS} topics")
print(f"📁 Output: {OUTPUT_FILE}")
