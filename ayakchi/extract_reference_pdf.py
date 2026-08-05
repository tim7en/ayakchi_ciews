"""Extract the user-supplied ADB project PDF to a reviewable text file."""

from pathlib import Path

from pypdf import PdfReader


SOURCE = Path(r"C:\Users\User\Documents\57055-001-tacr-en_2.pdf")
TARGET = Path(__file__).resolve().parent / "reference_57055-001-tacr-en_2.txt"


def main():
    reader = PdfReader(SOURCE)
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        sections.append(f"\n\n===== PDF PAGE {page_number} =====\n\n{page.extract_text() or ''}")
    TARGET.write_text("".join(sections), encoding="utf-8")
    print(f"Extracted {len(reader.pages)} pages to {TARGET}")


if __name__ == "__main__":
    main()
