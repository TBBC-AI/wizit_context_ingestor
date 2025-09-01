from pydantic import BaseModel, Field
from typing import List, Optional
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

# IMAGE_TRANSCRIPTION_SYSTEM_PROMPT = """
# Extract the content from an image page and output in Markdown syntax. Enclose the content in the <markdown></markdown> tag and do not use code blocks. If the image is empty then output a <markdown></markdown> without anything in it.

# Follow these steps:

# 1. Examine the provided page carefully.

# 2. Identify all elements present in the page, including headers, body text, footnotes, tables, images, captions, and page numbers, etc.

# 3. Use markdown syntax to format your output:
#     - Headings: # for main, ## for sections, ### for subsections, etc.
#     - Lists: * or - for bulleted, 1. 2. 3. for numbered
#     - Do not repeat yourself

# 4. If the element is an image (not table)
#     - If the information in the image can be represented by a table, generate the table containing the information of the image
#     - Otherwise provide a detailed description about the information in image
#     - Classify the element as one of: Chart, Diagram, Logo, Icon, Natural Image, Screenshot, Other. Enclose the class in <figure_type></figure_type>
#     - Enclose <figure_type></figure_type>, the table or description, and the figure title or caption (if available), in <figure></figure> tags
#     - Transcribe text in the image only after providing the table or description

# 5. If the element is a table
#     - Create a markdown table, ensuring every row has the same number of columns
#     - Maintain cell alignment as closely as possible
#     - Do not split a table into multiple tables
#     - If a merged cell spans multiple rows or columns, place the text in the top-left cell and output ' ' for other
#     - Use | for column separators, |-|-| for header row separators
#     - If a cell has multiple items, list them in separate rows
#     - If the table contains sub-headers, separate the sub-headers from the headers in another row

# 6. If the element is a paragraph
#     - Transcribe each text element precisely as it appears

# 7. If the element is a header, footer, footnote, page number
#     - Transcribe each text element precisely as it appears

# Output Example:
# <markdown>
# <figure>
# <figure_type>Chart</figure_type>
# Figure 3: This chart shows annual sales in millions. The year 2020 was significantly down due to the COVID-19 pandemic.
# A bar chart showing annual sales figures, with the y-axis labeled "Sales ($Million)" and the x-axis labeled "Year". The chart has bars for 2018 ($12M), 2019 ($18M), 2020 ($8M), and 2021 ($22M).
# </figure>

# <figure>
# <figure_type>Chart</figure_type>
# Figure 3: This chart shows annual sales in millions. The year 2020 was significantly down due to the COVID-19 pandemic.
# | Year | Sales ($Million) |
# |-|-|
# | 2018 | $12M |
# | 2019 | $18M |
# | 2020 | $8M |
# | 2021 | $22M |
# </figure>

# # Annual Report

# ## Financial Highlights

# <figure>
# <figure_type>Logo</figure_type>
# The logo of Apple Inc.
# </figure>

# * Revenue: $40M
# * Profit: $12M
# * EPS: $1.25

# | | Year Ended December 31, | |
# | | 2021 | 2022 |
# |-|-|-|
# | Cash provided by (used in): | | |
# | Operating activities | $ 46,327 | $ 46,752 |
# | Investing activities | (58,154) | (37,601) |
# | Financing activities | 6,291 | 9,718 |

# </markdown>
# """

# IMAGE_TRANSCRIPTION_SYSTEM_PROMPT = """
#     Please transcribe the exact text from the provided Document, regardless of length, ensuring extreme accuracy.
#     Never modify or summarize the text, just transcribe it.
#     Never include blank lines in the transcription.
#     It is essential to capture every piece of text exactly as it appears on each page, maintaining the original formatting and structure as closely as possible.
#     This includes headings, paragraphs, lists, tables, indents, and any text within images, with special attention to retain bold, italicized, or underlined formatting.
#     Your transcription must use Markdown and retain original formatting: Keep the layout of each page intact. This includes headings, paragraphs, lists, tables, indents, etc., noting any bold, italicized, or underlined text.
#     Handle Special Content: For tables, describe the layout and transcribe content cell by cell.
#     For images with text: provide a complete description of the image and transcribe the text within.
#     For tables: extract as many information as you can, provide a complete description of the table.
#     Make sure to transcribe any abbreviations or letter-number codes. Deal with Uncertainties: Mark unclear or illegible text as [unclear] or [illegible], providing a best guess where possible.
#     Capture All Text Types: Transcribe all text, whether in paragraphs, bullet points, captions under images, or within diagrams.
#     The goal is to complete the document's transcription, avoiding partial transcriptions.
#     Feedback and Error Reporting: Should you encounter issues that prevent the transcription of any page, please provide feedback on the nature of these issues and continue with the transcription of the following pages.
#     For each page/section/paragraph add a context heading and a brief description of the section to optimize the document for RAG (retrieval augmented generation)
#     ALWAYS USE THE SAME LANGUAGE OF THE DOCUMENT TO GENERATE THE CONTEXT HEADING AND DESCRIPTION
# """

SYNTHETIC_QUESTIONS_AND_RESPONSES_SYSTEM_PROMPT = """
    You are a helpful assistant that generates synthetic questions and responses from a given markdown content.
    <markdown_content>
    {markdown_content}
    </markdown_content>
    The questions and responses should be synthetic and cover all the topics in the markdown content.
    The questions and responses should be in the same language as the markdown content.
    NEVER ALTER THE OUTPUT FORMAT.
    Finally, output MUST be in the following format:
    {format_instructions}
"""

class QuestionAndResponse(BaseModel):
    question: str = Field(description="The question")
    response: str = Field(description="The response to the question")

class QuestionsAndResponses(BaseModel):
    items: List[QuestionAndResponse] = Field(description="List of questions and responses")

class ContextChunk(BaseModel):
    context: str = Field(description="Context description that helps with search retrieval")
    # chunk_keywords: str = Field(description="Key terms that aid search retrieval")
    # chunk_description: str = Field(description="What the chunk contains")
    # chunk_function: str = Field(description="The chunk's purpose (e.g., definition, example, instruction, list)")
    # chunk_structure: str = Field(description="Format type (paragraph, section, code block, etc.)")
    # chunk_main_idea: str = Field(description="Core concept or message")
