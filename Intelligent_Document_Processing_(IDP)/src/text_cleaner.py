import re


class TextProcessor:

    def clean_text(self, text: str) -> str:
        """
        Clean input text by:
        - Removing extra spaces
        - Removing unwanted special characters
        """
        text = re.sub(r'\s+', ' ', text)  # normalize whitespace
        text = re.sub(r'[^\w\s@.₹:-]', '', text)  # keep useful chars
        return text.strip()

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 200,
        overlap: int = 50,
        debug: bool = False
    ):
        """
        Chunk text with overlap (character-based)

        Args:
            text (str): input text
            chunk_size (int): size of each chunk
            overlap (int): overlapping characters between chunks
            debug (bool): print debug logs

        Returns:
            List[str]: list of text chunks
        """

        # ✅ Validation (prevents infinite loop)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            print("⚠ Overlap too large, adjusting automatically...")
            overlap = chunk_size // 5  # safe fallback

        chunks = []
        start = 0
        text_length = len(text)

        # ✅ Safety guard
        prev_start = -1

        while start < text_length:

            if start == prev_start:
                print("⚠ Infinite loop detected! Breaking...")
                break

            prev_start = start

            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            if debug:
                print(f"Chunk {len(chunks)}: start={start}, end={end}")

            # Move forward with overlap
            start += (chunk_size - overlap)

        return chunks


# ✅ Test block
if __name__ == "__main__":
    sample_text = "This is a sample text for chunking demonstration. " * 20

    processor = TextProcessor()

    # Step 1: Clean text
    clean = processor.clean_text(sample_text)

    # Step 2: Chunk text
    chunks = processor.chunk_text(
        clean,
        chunk_size=100,
        overlap=20,
        debug=True
    )

    print("\n✅ Final Chunks:\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}\n")