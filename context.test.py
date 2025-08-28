import json
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from wizit_context_ingestor.infra.vertex_model import VertexModels
from wizit_context_ingestor.infra.persistence import S3StorageService, LocalStorageService, SupabaseStoreService

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_PROJECT_LOCATION ="us-central1"# os.environ.get("GCP_PROJECT_LOCATION")


LLM_AS_JUDGE_SYSTEM_PROMPT = """
You are a language model evaluator. Your task is to assess how well a given chunk of text provides
meaningful and relevant context in relation to a specific query, based on the Given a document.

<markdown_content>
    {markdown_content}
</markdown_content>

Inputs:
Chunk: <chunk></chunk>

Context:<context ></context>

Instructions:
Compare the chunk and the context with the reference document.

Evaluate whether the chunk accurately and sufficiently supports the context.

Consider relevance, completeness, coherence, and faithfulness to the reference.

Output:

```json
Context Quality Score (0-5):

Reasoning: Briefly explain why you gave this score.

Suggestions for Improvement (if any):
```"""

def evaluate_chunk_generator(md_file_name: str, json_file_name: str):
    gcp_sa_path = os.path.join(os.path.dirname(__file__), "credentials", "gcp_sa.json")
    file_key = os.path.join(os.path.dirname(__file__), "tmp", json_file_name)
    markdown_file_key = os.path.join(os.path.dirname(__file__), "tmp", md_file_name)
    gcp_sa = None
    local_persistence_service = LocalStorageService()
    with open(gcp_sa_path, 'r') as gcp_sa_json:
        gcp_sa = json.loads(gcp_sa_json.read())
    vertex_model = VertexModels(
        GCP_PROJECT_ID,
        GCP_PROJECT_LOCATION,
        gcp_sa
    )

    print("loading chat model", file_key)

    vertex_model.load_chat_model()
    chunks_and_context = []
    with open(file_key, 'r') as file_json:
        chunks_and_context = json.loads(file_json.read())

    markdown_content = local_persistence_service.load_markdown_file_content(markdown_file_key)

    responses = []


    for chunk in chunks_and_context:
        # Create the prompt template with image
        prompt = ChatPromptTemplate.from_messages([
                    ("system", LLM_AS_JUDGE_SYSTEM_PROMPT),
                    (
                        "human", [{
                            "type": "text",
                                "text": f"""Evalutate context for the following chunk: <chunk>{chunk["page_content"]}</chunk>
                                <context>{chunk["metadata"]['context']}</context>"""
                        }]
                    ),
                ]).partial(
                    markdown_content=markdown_content,
                )
                # Create the chain
        chain = prompt | vertex_model.llm_model
                # Process the image
        results = chain.invoke({})
        responses.append(results.content)

    print(responses)


if __name__ == '__main__':
    # cloud_transcribe_document("TBBC-2025.pdf")
    # transcribe_document("infografias.pdf")
    # generate_synthetic_data("output.md")
    evaluate_chunk_generator("MyDoc.pdf.md", "context_chunks.json")
