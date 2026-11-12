
import json

from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class Generator:

    def __init__(
        self,
        model="llama3.2:3b",
        api_url="http://localhost:11434/api/generate",
        temperature=0.1,
        timeout=180
    ):

        self.model = model

        self.api_url = api_url
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, query, context):

        prompt = f"""
You are Stitch, an enterprise document intelligence
and knowledge retrieval assistant.

Your task is to answer questions using ONLY the
provided document context.

IMPORTANT ANSWERING RULES:

1. Use only information explicitly supported by the
   supplied document context.

2. Understand the meaning of the retrieved passages
   before answering.

3. Distinguish carefully between:
   - Definitions
   - Examples
   - Causes
   - Effects or consequences
   - Recommendations
   - Strategies
   - Headings and section titles

4. Do not interpret a heading, label, example, or
   unrelated statement as the answer to a definition question.

5. Do not combine unrelated sentences to create a
   definition or conclusion that the documents do not support.

6. If a retrieved passage is incomplete or starts or
   ends in the middle of a sentence, do not guess the
   missing information.

7. If the documents do not contain sufficient evidence
   to answer the question, explicitly say:
   "The information is unavailable in the provided documents."

8. Do not use your general knowledge to fill gaps.

9. Keep the answer clear, relevant, and concise.

10. Cite supporting information using the exact source
    filename and page number provided in the context.

11. Never invent source filenames or page numbers.

12. If multiple documents support the answer, cite
    each relevant source.

ANSWER FORMAT:

- Start with a direct answer.
- Explain briefly using supporting document information.
- Add source citations where applicable.
- Do not include irrelevant information.

DOCUMENT CONTEXT:

{context}

QUESTION:

{query}

ANSWER:
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        request = Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        try:

            with urlopen(request, timeout=self.timeout) as response:

                result = json.loads(
                    response.read().decode("utf-8")
                )

            return result["response"].strip()

        except HTTPError as error:

            raise RuntimeError(
                f"Ollama API returned HTTP {error.code}. "
                "Check that the model is available."
            ) from error

        except URLError as error:

            raise RuntimeError(
                "Cannot connect to Ollama. Open the Ollama app "
                "and make sure its local service is running."
            ) from error
