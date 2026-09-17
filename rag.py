import json
import os
import time
import uuid
from pathlib import Path

import faiss
import fitz
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class RAGEngine:
    def __init__(
        self,
        persist_directory="faiss_db",
        db_path=None,
        collection_name="research_papers",
    ):
        self.persist_directory = (
            persist_directory or db_path or "faiss_db"
        )

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------
        # Gemini configuration
        # ---------------------------------------------------------
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. "
                "Add it to your .env file."
            )

        self.primary_model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )

        self.backup_model = os.getenv(
            "GEMINI_BACKUP_MODEL",
            "gemini-3.7-flash",
        )

        self.embedding_model = os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-001",
        )

        # IMPORTANT:
        # Gemini embedding model supports 768 dimensions.
        self.embedding_dimension = 768

        self.top_k = int(
            os.getenv("TOP_K", "6")
        )

        self.chunk_size = int(
            os.getenv("CHUNK_SIZE", "1200")
        )

        self.chunk_overlap = int(
            os.getenv("CHUNK_OVERLAP", "180")
        )

        self.min_similarity = float(
            os.getenv("MIN_SIMILARITY", "0.22")
        )

        self.client = genai.Client(
            api_key=self.api_key
        )

        # ---------------------------------------------------------
        # FAISS + metadata
        # ---------------------------------------------------------
        self.index_path = (
            Path(self.persist_directory)
            / "research_papers.index"
        )

        self.metadata_path = (
            Path(self.persist_directory)
            / "metadata.json"
        )

        self.index = None
        self.metadata = []

        self._load_index()

    # =============================================================
    # FAISS STORAGE
    # =============================================================

    def _create_index(self):
        """
        Inner-product FAISS index.
        Vectors are normalized, so inner product ~= cosine similarity.
        """
        return faiss.IndexFlatIP(
            self.embedding_dimension
        )

    def _load_index(self):
        if (
            self.index_path.exists()
            and self.metadata_path.exists()
        ):
            try:
                self.index = faiss.read_index(
                    str(self.index_path)
                )

                with open(
                    self.metadata_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    self.metadata = json.load(file)

                # Safety check for dimension consistency.
                if (
                    self.index.d
                    != self.embedding_dimension
                ):
                    print(
                        "[FAISS] Existing index has "
                        f"{self.index.d} dimensions. "
                        f"Expected {self.embedding_dimension}."
                    )

                    self.index = self._create_index()
                    self.metadata = []

            except Exception as exc:
                print(
                    f"[FAISS] Could not load existing index: {exc}"
                )

                self.index = self._create_index()
                self.metadata = []

        else:
            self.index = self._create_index()
            self.metadata = []

    def _save_index(self):
        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

    # =============================================================
    # PDF EXTRACTION
    # =============================================================

    def extract_pdf_pages(self, pdf_path):
        pages = []

        document = fitz.open(pdf_path)

        try:
            for page_number, page in enumerate(
                document,
                start=1,
            ):
                text = page.get_text(
                    "text"
                ).strip()

                if text:
                    pages.append(
                        {
                            "page": page_number,
                            "text": text,
                        }
                    )
        finally:
            document.close()

        return pages

    # =============================================================
    # TEXT CHUNKING
    # =============================================================

    def split_text(self, text):
        words = text.split()

        if not words:
            return []

        chunks = []

        start = 0
        total_words = len(words)

        while start < total_words:
            end = min(
                start + self.chunk_size,
                total_words,
            )

            chunk = " ".join(
                words[start:end]
            ).strip()

            if chunk:
                chunks.append(chunk)

            if end >= total_words:
                break

            next_start = (
                end - self.chunk_overlap
            )

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    # =============================================================
    # GEMINI EMBEDDING
    # =============================================================

    def create_embedding(self, text):
        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.embedding_dimension,
            ),
        )

        vector = np.array(
            response.embeddings[0].values,
            dtype="float32",
        )

        if len(vector) != self.embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {self.embedding_dimension}, "
                f"got {len(vector)}."
            )

        # Normalize for cosine similarity using
        # FAISS inner-product search.
        faiss.normalize_L2(
            vector.reshape(1, -1)
        )

        return vector

    # =============================================================
    # INDEX PDF
    # =============================================================

    def index_pdf(self, pdf_path, display_name=None):
        pdf_path = str(pdf_path)

        filename = display_name or Path(pdf_path).name

        pages = self.extract_pdf_pages(
            pdf_path
        )

        if not pages:
            raise ValueError(
                "No readable text was found in this PDF."
            )

        document_id = str(
            uuid.uuid4()
        )

        vectors = []
        new_metadata = []

        chunk_number = 0

        for page_data in pages:
            page_number = page_data["page"]
            page_text = page_data["text"]

            chunks = self.split_text(
                page_text
            )

            for chunk in chunks:
                chunk_number += 1

                vector = self.create_embedding(
                    chunk
                )

                vectors.append(vector)

                new_metadata.append(
                    {
                        "document_id": document_id,
                        "document": filename,
                        "page": page_number,
                        "chunk": chunk_number,
                        "text": chunk,
                    }
                )

        if not vectors:
            raise ValueError(
                "No text chunks were generated from this PDF."
            )

        matrix = np.vstack(vectors).astype(
            "float32"
        )

        # Extra safety check.
        if matrix.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"FAISS expected {self.embedding_dimension} "
                f"dimensions but received "
                f"{matrix.shape[1]}."
            )

        self.index.add(matrix)

        self.metadata.extend(
            new_metadata
        )

        self._save_index()

        return {
            "document_id": document_id,
            "name": filename,
            "pages": len(pages),
            "chunks": len(new_metadata),
        }

    # =============================================================
    # DOCUMENT LIST
    # =============================================================

    def list_documents(self):
        documents = {}

        for item in self.metadata:
            document_id = item.get(
                "document_id"
            )

            if not document_id:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "id": document_id,
                    "name": item.get(
                        "document",
                        "Unknown document",
                    ),
                    "pages": set(),
                    "chunks": 0,
                }

            page = item.get("page")

            if page is not None:
                documents[
                    document_id
                ]["pages"].add(page)

            documents[
                document_id
            ]["chunks"] += 1

        output = []

        for document in documents.values():
            output.append(
                {
                    "id": document["id"],
                    "name": document["name"],
                    "pages": len(
                        document["pages"]
                    ),
                    "chunks": document["chunks"],
                }
            )

        output.sort(
            key=lambda item: item[
                "name"
            ].lower()
        )

        return output

    # =============================================================
    # STATUS
    # =============================================================

    def count(self):
        if self.index is None:
            return 0

        return int(
            self.index.ntotal
        )

    def chunk_count(self):
        return self.count()

    def has_documents(self):
        return self.count() > 0

    def stats(self):
        return {
            "papers": len(
                self.list_documents()
            ),
            "chunks": self.count(),
        }

    # =============================================================
    # SEARCH
    # =============================================================

    def search(self, question):
        if not self.has_documents():
            return []

        question_vector = self.create_embedding(
            question
        )

        query = question_vector.reshape(
            1, -1
        ).astype("float32")

        scores, indices = self.index.search(
            query,
            min(
                self.top_k,
                self.index.ntotal,
            ),
        )

        sources = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            similarity = float(score)

            if (
                similarity
                < self.min_similarity
            ):
                continue

            if index_position >= len(
                self.metadata
            ):
                continue

            item = self.metadata[
                index_position
            ]

            sources.append(
                {
                    "text": item.get(
                        "text",
                        "",
                    ),
                    "document": item.get(
                        "document",
                        "Unknown document",
                    ),
                    "page": item.get(
                        "page",
                        "?",
                    ),
                    "similarity": round(
                        max(
                            0.0,
                            min(
                                1.0,
                                similarity,
                            ),
                        ),
                        3,
                    ),
                }
            )

        return sources

    # =============================================================
    # GEMINI GENERATION
    # =============================================================

    def _generate_with_model(
        self,
        model,
        prompt,
        max_retries=2,
    ):
        last_error = None

        for attempt in range(
            max_retries + 1
        ):
            try:
                response = (
                    self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                )

                text = getattr(
                    response,
                    "text",
                    None,
                )

                if text and text.strip():
                    return text.strip()

                raise RuntimeError(
                    f"Gemini returned an empty "
                    f"response from {model}."
                )

            except Exception as exc:
                last_error = exc

                error_text = str(exc)

                # Only retry temporary API problems.
                is_transient = any(
                    code in error_text
                    for code in [
                        "503",
                        "UNAVAILABLE",
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "500",
                        "INTERNAL",
                        "408",
                        "TIMEOUT",
                    ]
                )

                if not is_transient:
                    raise

                if attempt < max_retries:
                    delay = 2 ** attempt

                    print(
                        f"[Gemini] {model} unavailable. "
                        f"Retrying in {delay}s..."
                    )

                    time.sleep(delay)

        raise RuntimeError(
            f"Model {model} failed after retries: "
            f"{last_error}"
        )

    def _generate_answer(self, prompt):
        models = []

        if self.primary_model:
            models.append(
                self.primary_model
            )

        if (
            self.backup_model
            and self.backup_model
            != self.primary_model
        ):
            models.append(
                self.backup_model
            )

        errors = []

        for model in models:
            try:
                print(
                    f"[Gemini] Trying model: {model}"
                )

                answer = (
                    self._generate_with_model(
                        model,
                        prompt,
                        max_retries=2,
                    )
                )

                print(
                    f"[Gemini] Answer generated "
                    f"using: {model}"
                )

                return answer

            except Exception as exc:
                errors.append(
                    f"{model}: {exc}"
                )

                print(
                    f"[Gemini] {model} failed: {exc}"
                )

        raise RuntimeError(
            "All Gemini generation models failed.\n"
            + "\n".join(errors)
        )

    # =============================================================
    # ANSWER QUESTION
    # =============================================================

    def answer(self, question):
        question = question.strip()

        if not question:
            raise ValueError(
                "Please enter a question."
            )

        if not self.has_documents():
            raise ValueError(
                "Please upload at least one research paper first."
            )

        sources = self.search(
            question
        )

        if not sources:
            return {
                "answer": (
                    "I could not find enough relevant "
                    "information in the uploaded research "
                    "papers to answer this question."
                ),
                "sources": [],
            }

        context_parts = []

        for number, source in enumerate(
            sources,
            start=1,
        ):
            context_parts.append(
                f"""
SOURCE {number}
Document: {source["document"]}
Page: {source["page"]}

{source["text"]}
""".strip()
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        prompt = f"""
You are a Research Paper Assistant.

Answer the user's question ONLY using the
uploaded research-paper context below.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the
   uploaded papers, clearly say so.
4. Give a clear and concise answer.
5. Mention the relevant paper and page when useful.
6. Every important claim must be supported by
   the provided context.

USER QUESTION:
{question}

UPLOADED PAPER CONTEXT:
{context}

Answer the question now using ONLY the
uploaded-paper context.
""".strip()

        try:
            answer = self._generate_answer(
                prompt
            )

        except Exception as exc:
            raise RuntimeError(
                f"Gemini generation failed: {exc}"
            )

        return {
            "answer": answer,
            "sources": [
                {
                    "document": source[
                        "document"
                    ],
                    "page": source["page"],
                    "similarity": source[
                        "similarity"
                    ],
                }
                for source in sources
            ],
        }

    # =============================================================
    # CLEAR ALL DOCUMENTS
    # =============================================================

    def clear(self):
        self.index = self._create_index()
        self.metadata = []

        self._save_index()