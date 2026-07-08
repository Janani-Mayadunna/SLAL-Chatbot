from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re

VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
TOP_K = 2
SEMANTIC_K = 12
FALLBACK_RESPONSE = "The information is unavailable in the Sri Lankan Airlines knowledge base."
GUARDRAIL_RESPONSE = "This chatbot answers only Sri Lankan Airlines policy questions."
TOXICITY_RESPONSE = "Please ask your question in respectful language."
UNKNOWN_SOURCE = "Unknown source"
MIN_RERANK_SCORE = 4
MIN_SEMANTIC_SIMILARITY = 0.38
SECTION_REF_PATTERN = re.compile(
    r"(?:^|(?<=\s))\d{1,2}(?:\.\d{1,2}){1,3}\s+(?=[A-Za-z])",
    re.MULTILINE,
)
INJECTION_PATTERNS = (
    re.compile(r"\bignore\b.{0,30}\b(previous|above|earlier|prior)\b.{0,20}\b(instruction|prompt|message)s?\b", re.IGNORECASE),
    re.compile(r"\bforget\b.{0,30}\b(previous|above|earlier|prior)\b.{0,20}\b(instruction|prompt|message)s?\b", re.IGNORECASE),
    re.compile(r"\b(act|behave|pretend)\b.{0,20}\bas\b", re.IGNORECASE),
    re.compile(r"\b(system prompt|developer message|hidden prompt|internal instruction)s?\b", re.IGNORECASE),
    re.compile(r"\b(reveal|show|tell|print|expose)\b.{0,20}\b(system prompt|developer message|instructions?)\b", re.IGNORECASE),
)
TOXIC_PATTERNS = (
    re.compile(r"\b(fuck|fucking|shit|bitch|bastard|asshole|moron|idiot|stupid)\b", re.IGNORECASE),
    re.compile(r"\b(hate you|kill yourself|shut up|piece of shit)\b", re.IGNORECASE),
)

PROMPT_TEMPLATE = """You are a Sri Lankan Airlines policy assistant.
Answer the question using only the policy text below.
Write clear sentences in plain English.
If the answer covers multiple classes, allowances, or policy sections, put each on its own line.
Do not repeat headings, labels, or section titles.
If the policy text does not answer the question, reply exactly:
I couldn't find this information in the knowledge base.

Policy text:
{evidence}

Question:
{question}

Answer:
"""

LABEL_PATTERN = re.compile(
    r"\b(?:Title|Section|Excerpt|Source|Description)\s*:\s*",
    re.IGNORECASE,
)
ANSWER_BREAK_PATTERN = re.compile(
    r"(?<!^)\s+(?=(?:"
    r"Business Class|"
    r"Economy Class|"
    r"Premium Economy(?: Class)?|"
    r"First Class|"
    r"Infant(?:\s*\([^)]*\))?|"
    r"\d{1,2}(?:\.\d{1,2})+\s+"
    r"))",
    re.IGNORECASE,
)


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    return tokenizer, model


def is_blocked_prompt(question):
    normalized = re.sub(r"\s+", " ", question.strip())
    return any(pattern.search(normalized) for pattern in INJECTION_PATTERNS)


def is_toxic_input(question):
    normalized = re.sub(r"\s+", " ", question.strip())
    return any(pattern.search(normalized) for pattern in TOXIC_PATTERNS)


def extract_keywords(question):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
    stop_words = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "does", "do", "did", "can", "could", "should", "would", "will",
        "the", "and", "for", "are", "with", "from", "that", "this",
        "have", "has", "had", "during", "into", "your", "their", "there",
        "Sri Lankan", "airlines", "please", "tell", "about", "policy", "offer",
        "passenger", "passengers",
    }
    return [word for word in words if word not in stop_words]


def extract_phrases(question):
    lowered = question.lower()
    phrases = []
    for match in re.finditer(r"[a-z]+(?:-[a-z]+)+", lowered):
        phrases.append(match.group(0))
    keywords = extract_keywords(question)
    phrases.extend(
        f"{keywords[i]} {keywords[i + 1]}"
        for i in range(len(keywords) - 1)
    )
    return phrases


def normalize_word(word):
    normalized = word.lower()
    if normalized.endswith("ies") and len(normalized) > 4:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("ing") and len(normalized) > 5:
        normalized = normalized[:-3]
    elif normalized.endswith("ed") and len(normalized) > 4:
        normalized = normalized[:-2]
    elif normalized.endswith("s") and len(normalized) > 4:
        normalized = normalized[:-1]
    if len(normalized) > 2 and normalized[-1] == normalized[-2]:
        normalized = normalized[:-1]
    return normalized


def tokenize_text(text):
    return [normalize_word(token) for token in re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())]


def policy_body(text):
    # Strip metadata labels so scoring focuses on the policy content itself.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Title:", "Source:", "Description:", "Section:")):
            continue
        if "====" in stripped or stripped == "END OF DOCUMENT":
            continue
        lines.append(stripped)
    return " ".join(lines)


def score_document(question, doc):
    # Combine lexical overlap with a few light reranking boosts.
    keywords = {normalize_word(w) for w in extract_keywords(question)}
    phrases = extract_phrases(question)
    body = policy_body(doc.page_content)
    body_lower = body.lower()
    body_tokens = set(tokenize_text(body))
    heading = doc.metadata.get("section_heading", "").lower()

    score = sum(len(k) for k in keywords if k in body_tokens)
    score += 5 * sum(1 for p in phrases if p in body_lower)

    title = doc.metadata.get("title", "")
    meta_tokens = set(tokenize_text(f"{title} {heading}"))
    score += 6 * len(keywords & meta_tokens)

    question_lower = question.lower()
    if "not allowed" in question_lower or "not permitted" in question_lower:
        if any(w in heading for w in ("restricted", "prohibited", "unacceptable")):
            score += 20
        if any(w in body_lower for w in ("prohibited", "must not", "unacceptable", "not be permitted")):
            score += 12
        if "carry-on" in question_lower and "allowance" in heading:
            score -= 15

    if "policy" in question_lower and "general" in heading:
        score += 12

    if "check-in" in question_lower or "boarding" in question_lower:
        if "check-in" in heading or "boarding" in heading:
            score += 15

    return score


def select_documents(question, vector_store):
    candidates = vector_store.similarity_search(question, k=SEMANTIC_K)
    if not candidates:
        return [], 0

    # Deduplicate sections before reranking so one section is not repeated.
    seen = set()
    unique_candidates = []
    for doc in candidates:
        key = (doc.metadata.get("source"), doc.metadata.get("section_id"))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(doc)

    ranked = sorted(
        ((score_document(question, doc), doc) for doc in unique_candidates),
        key=lambda item: item[0],
        reverse=True,
    )

    if not ranked or ranked[0][0] < MIN_RERANK_SCORE:
        return [], ranked[0][0] if ranked else 0

    best_score = ranked[0][0]
    selected = [
        doc for score, doc in ranked
        if score >= max(best_score - 2, MIN_RERANK_SCORE)
    ][:TOP_K]
    return selected, best_score


def split_sentences(text):
    parts = re.split(
        r"(?<=[.!?])\s+|(?<=;)\s+|(?<=:)\s+(?=\([a-z]\))|(?<=\))\s+(?=\([a-z]\))",
        text,
    )
    return [p.strip() for p in parts if p.strip()]


def best_excerpt(doc, question, max_sentences=3):
    body = policy_body(doc.page_content)
    keywords = {normalize_word(w) for w in extract_keywords(question)}
    phrases = extract_phrases(question)
    sentences = split_sentences(body)

    scored = []
    for idx, sentence in enumerate(sentences):
        tokens = set(tokenize_text(sentence))
        s = sum(len(k) for k in keywords if k in tokens)
        s += 5 * sum(1 for p in phrases if p in sentence.lower())
        if s > 0:
            scored.append((s, idx, sentence))
            if sentence.endswith(":") and idx + 1 < len(sentences):
                scored.append((s + 1, idx + 1, sentences[idx + 1]))

    if not scored:
        if keywords and not any(
            keyword_matches(kw, body.lower(), set(tokenize_text(body)))
            for kw in keywords
        ):
            return ""
        return body[:600]

    scored.sort(key=lambda item: (-item[0], item[1]))
    top = scored[:max_sentences]
    ordered = [sent for _, _, sent in sorted(top, key=lambda item: item[1])]
    return " ".join(ordered)


def build_evidence(documents, question):
    # Evidence is passed to the LLM as short grounded excerpts.
    excerpts = []
    for i, doc in enumerate(documents, start=1):
        excerpt = best_excerpt(doc, question)
        excerpts.append(f"[{i}] {excerpt}")
    return "\n\n".join(excerpts)


def clean_final_answer(answer):
    cleaned = answer.strip()
    cleaned = LABEL_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\[\d+\]\s*", "", cleaned)
    cleaned = re.sub(r"^\d{1,2}(?:\.\d{1,2}){1,3}\s+(?=[A-Za-z])", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    return cleaned.strip()


def format_answer_display(answer):
    if not answer or answer in {FALLBACK_RESPONSE, GUARDRAIL_RESPONSE, TOXICITY_RESPONSE}:
        return answer

    formatted = ANSWER_BREAK_PATTERN.sub("\n", answer.strip())
    lines = []
    for line in formatted.splitlines():
        line = line.strip()
        if not line:
            continue
        line = SECTION_REF_PATTERN.sub("", line).strip()
        if line:
            lines.append(line)
    return "\n\n".join(lines)


def capitalize_sentence(text):
    if not text:
        return text
    return text[0].upper() + text[1:]


def evidence_to_answer(evidence):
    text = LABEL_PATTERN.sub("", evidence)
    text = re.sub(r"\[\d+\]\s*", "", text)
    text = SECTION_REF_PATTERN.sub("", text)
    parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not parts:
        return FALLBACK_RESPONSE
    # Keep each retrieved excerpt on its own paragraph for readability.
    answer = "\n\n".join(parts[:2])
    answer = capitalize_sentence(answer)
    if answer and answer[-1] not in ".!?":
        answer += "."
    return answer


def is_unusable_answer(answer):
    cleaned = clean_final_answer(answer)
    if not cleaned or cleaned == FALLBACK_RESPONSE:
        return not cleaned
    if len(cleaned.split()) < 8:
        return True
    if cleaned.endswith(":"):
        return True
    if LABEL_PATTERN.search(cleaned):
        return True
    if re.search(r"\b(?:INVOLUNTARY|VOLUNTARY)\s+REFUNDS\b", cleaned):
        return True
    if cleaned.lower().startswith(("inform ", "obtain ", "acceptance of")):
        return True
    return False


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_similarity(question, evidence, vector_store):
    embeddings = vector_store.embedding_function
    question_vec = embeddings.embed_query(question)
    evidence_vec = embeddings.embed_query(evidence)
    return cosine_similarity(question_vec, evidence_vec)


def keyword_matches(keyword, evidence_lower, evidence_tokens):
    normalized = normalize_word(keyword)
    return normalized in evidence_tokens or normalized in evidence_lower


def evidence_supports_question(question, evidence, vector_store):
    # Gate answers so unrelated retrievals fall back instead of hallucinating.
    if semantic_similarity(question, evidence, vector_store) < MIN_SEMANTIC_SIMILARITY:
        return False
    return evidence_answers_question(question, evidence)


def evidence_answers_question(question, evidence):
    keywords = extract_keywords(question)
    if not keywords:
        return False

    evidence_lower = evidence.lower()
    evidence_tokens = set(tokenize_text(evidence))
    matches = [
        kw for kw in keywords
        if keyword_matches(kw, evidence_lower, evidence_tokens)
    ]

    if not matches:
        return False
    if len(keywords) == 1:
        return True
    if len(matches) >= 2:
        return True
    return len(matches) / len(keywords) >= 0.5


def generate_answer(question, evidence, llm, max_new_tokens=160):
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted = prompt.format(evidence=evidence, question=question)
    tokenizer, model = llm
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return clean_final_answer(tokenizer.decode(outputs[0], skip_special_tokens=True))


def answer_question(question, vector_store, llm):
    if is_toxic_input(question):
        return TOXICITY_RESPONSE, []

    # Block basic prompt-injection attempts before retrieval or generation.
    if is_blocked_prompt(question):
        return GUARDRAIL_RESPONSE, []

    docs, best_score = select_documents(question, vector_store)
    if not docs:
        return FALLBACK_RESPONSE, []

    evidence = build_evidence(docs, question)
    if not evidence.strip():
        return FALLBACK_RESPONSE, docs

    if not evidence_supports_question(question, evidence, vector_store):
        return FALLBACK_RESPONSE, []

    response = generate_answer(question, evidence, llm)
    if is_unusable_answer(response):
        response = generate_answer(question, evidence, llm, max_new_tokens=220)

    if is_unusable_answer(response):
        response = evidence_to_answer(evidence)

    response = clean_final_answer(response)
    response = capitalize_sentence(response)
    if response and response[-1] not in ".!?":
        response += "."

    if (
        response == FALLBACK_RESPONSE
        and best_score >= MIN_RERANK_SCORE
        and evidence_supports_question(question, evidence, vector_store)
    ):
        response = evidence_to_answer(evidence)

    response = format_answer_display(response)
    return response, docs


def main():
    vector_store = load_vector_store()
    llm = load_llm()
    print("Chatbot is ready. Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        answer, retrieved_docs = answer_question(question, vector_store, llm)

        print("\nAnswer:")
        print(answer)
        print("\nRetrieved sources:")
        seen = set()
        for doc in retrieved_docs:
            source = doc.metadata.get("source", UNKNOWN_SOURCE)
            heading = doc.metadata.get("section_heading", "")
            label = f"{source} ({heading})" if heading else source
            if label not in seen:
                print(f"- {label}")
                seen.add(label)
        print()


if __name__ == "__main__":
    main()
