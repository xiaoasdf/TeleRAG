from pathlib import Path
from typing import Dict, List, Union

from pypdf import PdfReader


def load_txt(file_path: Union[str, Path]) -> Dict:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    return {
        "text": text,
        "source": path.name,
        "doc_type": "txt",
        "pages": None,
    }


def load_md(file_path: Union[str, Path]) -> Dict:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    return {
        "text": text,
        "source": path.name,
        "doc_type": "md",
        "pages": None,
    }


def load_pdf(file_path: Union[str, Path]) -> Dict:
    path = Path(file_path)
    reader = PdfReader(str(path))

    page_texts: List[str] = []

    for page in reader.pages:
        text = page.extract_text()
        page_texts.append(text if text else "")

    full_text = "\n".join(page_texts)

    return {
        "text": full_text,
        "source": path.name,
        "doc_type": "pdf",
        "pages": len(reader.pages),
        "page_texts": page_texts,
    }


def load_document(file_path: Union[str, Path]) -> Dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_txt(path)
    elif suffix == ".md":
        return load_md(path)
    elif suffix == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")