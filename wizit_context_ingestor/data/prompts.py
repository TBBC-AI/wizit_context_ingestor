from pydantic import BaseModel, Field

IMAGE_TRANSCRIPTION_SYSTEM_PROMPT = """
Transcribe the exact text from the provided Document, regardless of length, ensuring extreme accuracy. Organize the transcript using markdown.
Follow these steps:
<steps>
1. Check every piece of content, then determine the main language of the document. This language will be used for the transcription of every piece of information.
2. Examine the provided page carefully. It is essential to capture every piece of text exactly as it appears on each page, maintaining the original mainlanguage, formatting and structure as closely as possible.
3. Identify all elements present in the page, including headings, body text, footnotes, tables, images, captions, page numbers, paragraphs, lists, indents, and any text within images, with special attention to retain bold, italicized, or underlined formatting, etc.
4. For images or figures with content (not tables):
    - Ensure you retrieve meaningful descriptions of the image content.
    - If the information in the image can be represented by a table, generate the table containing the information of the image, otherwise provide a detailed description about the information in the image
    - Classify the element as one of: Chart, Diagram, Natural Image, Screenshot, Other. Enclose the class in <figure_type></figure_type>
    - Enclose <figure_type></figure_type>, the table or description, and the figure title, caption (if available) or description in <figure></figure> tags
5. For tables:
    - Create a markdown table
    - Maintain cell alignment as closely as possible
    - Transcribe the table as well as possible
6. Check every piece of transcribed content and ensure it is in the main language of the document: content, figures, tables. When this condition is not met, provide a translation of the content in the main language.
7. Use markdown syntax to format your output:
    - Headings: # for main, ## for sections, ### for subsections, etc.
    - Lists: * or - for bulleted, 1. 2. 3. for numbered
</steps>
RULES:
<rules>
1. Transcribe all text exactly as it appears, including:
   - Paragraphs
   - Headers and footers
   - Footnotes and page numbers
   - Text in bullet points and lists
   - Captions under images
   - Text within diagrams
2. Mark unclear or illegible text as [unclear] or [illegible], providing a best guess where possible.
3. All transcribed content must be in the main document language.
4. Complete the entire document transcription - avoid partial transcriptions.
5. Never generate information by yourself, modify or summarize the text, only transcribe the text exactly as it appears.
6. Never include blank lines in the transcription.
7. Do not include logos or icons in your transcriptions
8. Do not include special characters or symbols that may interfere with markdown formatting.
9. Do not include encoded image content.
10. Do not transcribe logos, icons or watermarks.
11. Do not include any content in your transcription that is not in the same main language of the document.
</rules>
"""

CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT = """
    You are a helpful assistant that generates context chunks from a given markdown content.
    TASK:
    Think step by step:
    <task_analysis>
    1. Language Detection: Identify main language used in the document
    2. Context Generation: Create a brief context description that helps with search retrieval, your context must include all these elements within the text:
    - chunk_relation_with_document: How this chunk fits within the overall document
    - chunk_keywords: Key terms that aid search retrieval
    - chunk_description: What the chunk contains
    - chunk_function: The chunk's purpose (e.g., definition, example, instruction, list)
    - chunk_structure: Format type (paragraph, section, code block, etc.)
    - chunk_main_idea: Core concept or message
    3. The generated context must be in the same language of the document content
    </task_analysis>
    CRITICAL RULES:
    <critical_rules>
    - Context MUST be in the SAME language of the source document content
    - Be concise but informative
    - Focus on search retrieval optimization
    - Do NOT include the original chunk content
    </critical_rules>
    <document_content>
    {document_content}
    </document_content>
    Finally,:
    {format_instructions}
"""

class ContextChunk(BaseModel):
    context: str = Field(description="Context description that helps with search retrieval")
