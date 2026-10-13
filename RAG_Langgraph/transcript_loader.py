import os
import re
import tempfile
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Matches "00:00:04,000 --> 00:00:08,000" (.srt) or "00:00:04.000 --> 00:00:08.000" (.vtt)
# followed by caption text, up to the next blank line.
CUE_PATTERN = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*\n(.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def _parse_timestamp_cues(raw_text: str) -> list[tuple[str, str]]:
    """Extracts (start_timestamp, caption_text) pairs from .srt/.vtt content."""
    cues = []
    for match in CUE_PATTERN.finditer(raw_text):
        start_time = match.group(1).replace(",", ".")
        text = " ".join(line.strip() for line in match.group(2).splitlines() if line.strip())
        if text:
            cues.append((start_time, text))
    return cues


def _chunk_cues(cues: list[tuple[str, str]], source: str, chunk_chars: int = 1000) -> list[Document]:
    """
    Groups consecutive subtitle cues into ~chunk_chars sized Documents.
    Each chunk keeps the timestamp of its FIRST cue, so answers can cite
    roughly when in the video/meeting something was said.
    """
    chunks = []
    buffer_text, buffer_start, buffer_len = [], None, 0

    def flush():
        if buffer_text:
            chunks.append(Document(
                page_content=" ".join(buffer_text),
                metadata={"source": source, "start_time": buffer_start},
            ))

    for start_time, text in cues:
        if buffer_start is None:
            buffer_start = start_time
        buffer_text.append(text)
        buffer_len += len(text)
        if buffer_len >= chunk_chars:
            flush()
            buffer_text, buffer_start, buffer_len = [], None, 0

    flush()
    return chunks


def load_transcript(uploaded_file) -> tuple[list[Document], str]:
    """
    Takes a Streamlit UploadedFile (.txt, .pdf, .srt or .vtt) and returns:
      - chunks: list[Document] ready to be embedded into a vector store
      - full_text: the complete transcript as one string, for summarization
    """
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if extension == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
            for doc in docs:
                doc.metadata["source"] = filename
            full_text = "\n\n".join(doc.page_content for doc in docs)
            return SPLITTER.split_documents(docs), full_text

        if extension == ".txt":
            docs = TextLoader(tmp_path, encoding="utf-8").load()
            for doc in docs:
                doc.metadata["source"] = filename
            full_text = "\n\n".join(doc.page_content for doc in docs)
            return SPLITTER.split_documents(docs), full_text

        if extension in (".srt", ".vtt"):
            with open(tmp_path, encoding="utf-8") as f:
                raw_text = f.read()
            cues = _parse_timestamp_cues(raw_text)
            if not cues:
                raise ValueError(f"Could not find any timestamped cues in {filename}")
            full_text = " ".join(text for _, text in cues)
            return _chunk_cues(cues, source=filename), full_text

        raise ValueError(f"Unsupported transcript format: {extension}")
    finally:
        os.remove(tmp_path)
