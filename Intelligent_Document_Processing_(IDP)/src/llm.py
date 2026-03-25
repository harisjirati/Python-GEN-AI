import requests


class LLM:
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def generate_answer(self, query: str, context: str, detailed: bool = False) -> str:
        """
        Answer any question about a document using Llama.

        Two modes:
        - detailed=False (default): short, direct answer — one sentence or less
        - detailed=True: full explanation with reasoning

        The prompt is structured to work well with invoice/contract/report style docs.
        """

        if detailed:
            instruction = (
                "Give a thorough, detailed answer. "
                "Explain your reasoning. "
                "If relevant, mention related fields from the document."
            )
            max_tokens = 300
        else:
            instruction = (
                "Give a SHORT and DIRECT answer only. "
                "One sentence maximum. "
                "No explanation unless absolutely necessary."
            )
            max_tokens = 80

        prompt = f"""You are a document assistant. You answer questions based strictly on the document content provided below.

Rules:
- Answer ONLY from the context. Do not make up information.
- If the answer is not in the context, say: "This information is not in the provided documents."
- {instruction}

Document content:
{context}

Question: {query}

Answer:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,    # slight creativity for natural phrasing
                "top_p": 0.9,
                "num_predict": max_tokens,
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()

            answer = response.json().get("response", "").strip()

            # Clean up any trailing incomplete sentence on short answers
            if not detailed and "." in answer:
                answer = answer.split(".")[0].strip() + "."

            return answer if answer else "This information is not in the provided documents."

        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Start it with: ollama serve"
        except requests.exceptions.Timeout:
            return "❌ Ollama timeout. Try a shorter question."
        except Exception as e:
            return f"❌ LLM error: {str(e)}"

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return any(self.model in m for m in models)
            return False
        except Exception:
            return False