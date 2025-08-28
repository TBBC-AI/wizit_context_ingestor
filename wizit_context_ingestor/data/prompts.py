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
    Ensure that the transcription is accurate, complete and keeps the original language intact.
    For each page/section/paragraph add a context heading and a brief description of the section to optimize the document for RAG (retrieval augmented generation)
    ALWAYS USE THE SAME LANGUAGE OF THE DOCUMENT TO GENERATE THE CONTEXT HEADING AND DESCRIPTION
"""

# PARSE_DOC_SYSTEM_PROMPT = """
#     Please transcribe the exact text from the provided Document, regardless of length, ensuring extreme accuracy.
#     It is essential to capture every piece of text exactly as it appears on each page, maintaining the original formatting and structure as closely as possible.
#     This includes headings, paragraphs, lists, tables, indents, and any text within images, with special attention to retain bold, italicized, or underlined formatting.
#     Your transcription must use Markdown and retain original formatting: Keep the layout of each page intact. This includes headings, paragraphs, lists, tables, indents, etc., noting any bold, italicized, or underlined text.
#     Handle Special Content: For tables, describe the layout and transcribe content cell by cell.
#     For images with text: provide a complete description of the image and transcribe the text within.
#     For tables: extract as many information as you can, provide a complete description of the table.
#     Make sure to transcribe any abbreviations or letter-number codes. Deal with Uncertainties: Mark unclear or illegible text as [unclear] or [illegible], providing a best guess where possible.
#     Capture All Text Types: Transcribe all text, whether in paragraphs, bullet points, captions under images, or within diagrams.
#     Ensure Continuous Processing: The task requires processing each page sequentially until the entire document is transcribed.
#     If errors, unusual formats, or unclear text prevent accurate transcription of a page, note the issue and proceed to the next page.
#     The goal is to complete the document's transcription, avoiding partial transcriptions unless specified.
#     Feedback and Error Reporting: Should you encounter issues that prevent the transcription of any page, please provide feedback on the nature of these issues and continue with the transcription of the following pages.
# """

PARSE_DOC_HUMAN_PROMPT = """
    Now, transcribre the following document
"""
