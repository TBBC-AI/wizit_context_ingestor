from pydantic import BaseModel, Field

PARSE_DOC_SYSTEM_PROMPT = """
    Please transcribe the exact text from the provided Document, regardless of length, ensuring extreme accuracy.
    It is essential to capture every piece of text exactly as it appears on each page, maintaining the original formatting and structure as closely as possible.
    This includes headings, paragraphs, lists, tables, indents, and any text within images, with special attention to retain bold, italicized, or underlined formatting.
    Your transcription must use Markdown and retain original formatting: Keep the layout of each page intact. This includes headings, paragraphs, lists, tables, indents, etc., noting any bold, italicized, or underlined text.
    Handle Special Content: For tables, describe the layout and transcribe content cell by cell.
    For images with text: provide a complete description of the image and transcribe the text within.
    For tables: extract as many information as you can, provide a complete description of the table.
    Make sure to transcribe any abbreviations or letter-number codes. Deal with Uncertainties: Mark unclear or illegible text as [unclear] or [illegible], providing a best guess where possible.
    Capture All Text Types: Transcribe all text, whether in paragraphs, bullet points, captions under images, or within diagrams.
    Ensure Continuous Processing: The task requires processing each page sequentially until the entire document is transcribed.
    If errors, unusual formats, or unclear text prevent accurate transcription of a page, note the issue and proceed to the next page.
    The goal is to complete the document's transcription, avoiding partial transcriptions unless specified.
    Feedback and Error Reporting: Should you encounter issues that prevent the transcription of any page, please provide feedback on the nature of these issues and continue with the transcription of the following pages.
    For each page/section/paragraph add a context heading and a brief description of the section to optimize the document for RAG (retrieval augmented generation)
    ALWAYS USE THE SAME LANGUAGE OF THE DOCUMENT TO GENERATE THE CONTEXT HEADING AND DESCRIPTION
"""

IMAGE_TRANSCRIPTION_SYSTEM_PROMPT = """
Transcribe the exact text from the provided Document, regardless of length, ensuring extreme accuracy. Organize the transcript using markdown.

Follow these steps:

1. Examine the provided page carefully. It is essential to capture every piece of text exactly as it appears on each page, maintaining the original language,formatting and structure as closely as possible.
2. Identify all elements present in the page, including headings, body text, footnotes, tables, images, captions, page numbers, paragraphs, lists, indents, and any text within images, with special attention to retain bold, italicized, or underlined formatting, etc.
3. Use markdown syntax to format your output:
    - Headings: # for main, ## for sections, ### for subsections, etc.
    - Lists: * or - for bulleted, 1. 2. 3. for numbered

4. If the element is an image (not table)
    - If the information in the image can be represented by a table, generate the table containing the information of the image, otherwise provide a detailed description about the information in the image
    - Classify the element as one of: Chart, Diagram, Natural Image, Screenshot, Other. Enclose the class in <figure_type></figure_type>
    - Enclose <figure_type></figure_type>, the table or description, and the figure title or caption (if available), in <figure></figure> tags
    - Do not transcribe text in the image after providing the table or description
    - Do not include encoded image content.
    - Do not transcribe logos, icons or watermarks.

5. If the element is a table
    - Create a markdown table, ensuring every row has the same number of columns
    - Maintain cell alignment as closely as possible
    - Do not split a table into multiple tables
    - If a merged cell spans multiple rows or columns, place the text in the top-left cell and output ' ' for other
    - Use | for column separators, |-|-| for header row separators
    - If a cell has multiple items, list them in separate rows
    - If the table contains sub-headers, separate the sub-headers from the headers in another row

RULES:
1. Transcribe all text exactly as it appears, including:
   - Paragraphs
   - Headers and footers
   - Footnotes and page numbers
   - Text in bullet points and lists
   - Captions under images
   - Text within diagrams
2. Never modify or summarize the text, just transcribe it.
3. Mark unclear or illegible text as [unclear] or [illegible], providing a best guess where possible.
5. All generated content (transcription, context fields, descriptions) must be in the original document language.
6. Complete the entire document transcription - avoid partial transcriptions.
7. Never generate information by yourself, only transcribe the text exactly as it appears.
8. Never include blank lines in the transcription.
10. Do not include logos or icons in your transcriptions
"""

CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT = """
    You are a helpful assistant that generates context chunks from a given markdown content.
    TASK:
    Think step by step:
    <task_analysis>
    1. Language Detection: Identify the document content main language
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
