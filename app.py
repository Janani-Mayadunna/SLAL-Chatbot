import gradio as gr

from chatbot import answer_question, load_llm, load_vector_store

VECTOR_STORE = load_vector_store()
LLM = load_llm()


def format_sources(documents):
    if not documents:
        return "**Retrieved Sources**\n\nNo sources retrieved."

    seen_sources = {}
    for doc in documents:
        source = doc.metadata.get("source", "Unknown source")
        filename = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if filename in seen_sources:
            continue

        section = (
            doc.metadata.get("title", "").strip()
            or doc.metadata.get("section_heading", "").strip()
        )
        if section == "Unknown title":
            section = ""

        seen_sources[filename] = section

    lines = ["**Retrieved Sources**", ""]
    for filename, section in seen_sources.items():
        if section:
            lines.append(f"- {filename} - Section: {section}")
        else:
            lines.append(f"- {filename}")

    return "\n".join(lines)


def respond(question):
    question = question.strip()
    if not question:
        return "Please enter a question.", format_sources([])

    answer, retrieved_docs = answer_question(question, VECTOR_STORE, LLM)
    return answer, format_sources(retrieved_docs)


with gr.Blocks(title="Sri Lankan Airlines RAG Chatbot") as demo:
    gr.Markdown(
        """
        # Sri Lankan Airlines RAG Chatbot
        Ask questions about Sri Lankan Airlines Conditions of Carriage.

        This chatbot retrieves relevant policy text from the knowledge base and
        answers only from that retrieved information.
        """
    )

    question_input = gr.Textbox(
        label="Your Question",
        placeholder="Example: What is the refund policy?",
        lines=2,
    )

    ask_button = gr.Button("Ask")

    answer_output = gr.Textbox(
        label="Chatbot Answer",
        lines=5,
        interactive=False,
    )

    sources_output = gr.Markdown(label="Retrieved Sources")

    # Support both button click and Enter key submission.
    ask_button.click(fn=respond, inputs=question_input, outputs=[answer_output, sources_output])
    question_input.submit(fn=respond, inputs=question_input, outputs=[answer_output, sources_output])


if __name__ == "__main__":
    demo.launch()
