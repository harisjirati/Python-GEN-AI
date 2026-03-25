import pymupdf as fitz


class PDFReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_text(self) -> str:
        """
        Extract text from PDF file
        """
        text = ""

        try:
            # Open PDF
            doc = fitz.open(self.file_path)

            # Loop through pages
            for page in doc:
                # Extract text (sorted for better readability)
                text += page.get_text("text", sort=True)

            doc.close()
            return text

        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return ""


# Test the module directly
if __name__ == "__main__":
    pdf_path = "./data/sample.pdf"

    reader = PDFReader(pdf_path)
    extracted_text = reader.extract_text()

    print("\n📄 Extracted Text:\n")
    print(extracted_text)