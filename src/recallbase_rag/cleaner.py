import re
import unicodedata

from recallbase_rag.schemas import Page

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACES_AND_TABS = re.compile(r"[ \t]+")
SPACES_AROUND_NEWLINE = re.compile(r" *\n *")
HYPHEN_LINE_BREAK = re.compile(r"([a-z])-\n([a-z])")
MANY_NEWLINES = re.compile(r"\n{3,}")
PRIVATE_USE_CHARS = re.compile(r"[\ue000-\uf8ff]")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u00ad", "")
    text = CONTROL_CHARS.sub("", text)
    text = PRIVATE_USE_CHARS.sub(" ", text)
    text = SPACES_AND_TABS.sub(" ", text)
    text = SPACES_AROUND_NEWLINE.sub("\n", text)
    text = HYPHEN_LINE_BREAK.sub(r"\1\2", text)
    text = MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def clean_pages(pages: list[Page]) -> list[Page]:
    cleaned_pages: list[Page] = []

    for page in pages:
        cleaned_text = clean_text(page.text)
        cleaned_page = Page(
            file_name=page.file_name,
            page_number=page.page_number,
            text=cleaned_text,
        )
        cleaned_pages.append(cleaned_page)

    return cleaned_pages