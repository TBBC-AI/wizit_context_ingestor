from langchain_core.tools import tool


@tool(parse_docstring=True)
def complete_context_gen(context: str) -> str:
    """Tool to generate comprehensive contextual information for a document chunk.

    This tool creates enriched context by analyzing how a specific chunk relates to and fits
    within the broader document structure. Use this after you've identified the chunk's role,
    relationships, and significance within the document.

    When to use:
    - After analyzing a chunk's position and purpose within the overall document
    - When you need to establish connections between the chunk and surrounding content
    - Before finalizing context generation to ensure comprehensive understanding
    - When preparing detailed contextual information for downstream processing

    Analysis should address:
    1. Document integration - How does this chunk contribute to the document's main themes and objectives?
    2. Structural relationships - How does the chunk connect with preceding and following sections?
    3. Content dependencies - What key concepts, references, or information does this chunk rely on or provide?
    4. Semantic coherence - How does the chunk maintain consistency with the document's tone and message?

    Args:
        context: Your detailed analysis and contextual information for the document chunk. must use the same chunk language.

    Returns:
        The processed contextual information ready for use. must use the same chunk language.
    """
    return f"{context}"


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
