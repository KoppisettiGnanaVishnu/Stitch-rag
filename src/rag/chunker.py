
import re


def split_into_sentences(text):
    """
    Split text into sentences while preserving sentence punctuation.
    Also handle paragraph and line breaks.
    """

    sentences = re.split(
        r'(?<=[.!?])\s+|\n+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_long_sentence(sentence, chunk_size):
    """
    Split a sentence that exceeds chunk_size into smaller
    word-based pieces.
    """

    words = sentence.split()
    pieces = []
    current = ""

    for word in words:

        if len(word) > chunk_size:
            if current:
                pieces.append(current)
                current = ""

            for i in range(0, len(word), chunk_size):
                pieces.append(word[i:i + chunk_size])

            continue

        candidate = f"{current} {word}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current)

            current = word

    if current:
        pieces.append(current)

    return pieces


def create_chunks(documents, chunk_size=500, chunk_overlap=100):

    chunks = []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    for document in documents:

        text = document["text"].strip()

        source = document["source"]
        page = document["page"]

        if not text:
            continue

        # Step 1: Split document text into sentences
        sentences = split_into_sentences(text)

        # Step 2: Split exceptionally long sentences
        processed_sentences = []

        for sentence in sentences:

            if len(sentence) > chunk_size:
                pieces = split_long_sentence(
                    sentence,
                    chunk_size
                )

                processed_sentences.extend(pieces)

            else:
                processed_sentences.append(sentence)

        # Step 3: Combine sentences into chunks
        current_sentences = []

        for sentence in processed_sentences:

            candidate = " ".join(
                current_sentences + [sentence]
            )

            if (
                current_sentences
                and len(candidate) > chunk_size
            ):

                chunk_text = " ".join(
                    current_sentences
                ).strip()

                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page
                })

                # Step 4: Preserve sentence-based overlap
                overlap_sentences = []
                overlap_length = 0

                for previous_sentence in reversed(
                    current_sentences
                ):

                    required_length = (
                        len(previous_sentence)
                        + (1 if overlap_sentences else 0)
                    )

                    if (
                        overlap_length + required_length
                        > chunk_overlap
                    ):
                        break

                    overlap_sentences.insert(
                        0,
                        previous_sentence
                    )

                    overlap_length += required_length

                current_sentences = overlap_sentences

                # Remove overlap sentences if needed
                # to accommodate the next sentence
                while current_sentences and len(
                    " ".join(current_sentences + [sentence])
                ) > chunk_size:

                    current_sentences.pop(0)

            current_sentences.append(sentence)

        # Step 5: Add the remaining text
        if current_sentences:

            chunk_text = " ".join(
                current_sentences
            ).strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page
                })

    return chunks
