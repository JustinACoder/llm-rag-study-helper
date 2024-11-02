import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
import os
import re
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Initialize global variables
vector_store = None
qa_chain = None
default_prompt = """You are an AI assistant that helps answer questions based on a set of documents.
The documents you are provided are the outputs of a PDF reader and selected using a retrieval system.
Therefore, the quality of syntax and structure may vary. However, you should treat the information as the absolute truth.
The questions you will be asked are based on the content of these documents every time. 
Therefore, if you can't justify your answer with the documents, you should indicate that you can't answer the question because it's out of scope.
However, note that the answer could be a very short sentence in a sea of text or that it should be inferred from the context of the documents.
Therefore, you have to be very attentive to the details and the context of the documents.
If applicable, it is best to quote the relevant part of the document to justify your answer.
Given the context and the question, provide the answer and mention the relevant documents.

<context>
{context}
</context>

<question>
{input}
</question>"""


# Validate prompt
def validate_prompt(prompt_text):
    if "{context}" in prompt_text and "{input}" in prompt_text:
        return "Valid prompt. Ready to load documents.", True
    else:
        return "Invalid prompt. Please include {context} and {input}.", False


def get_validate_prompt_message(prompt_text):
    return validate_prompt(prompt_text)[0]


# Set API Key
def set_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    return "API Key set successfully."


# Preprocess Text
def preprocess_text(documents):
    combined_text = ""
    for doc in documents:
        combined_text += doc.page_content + "\n"

    # Remove unwanted sections and patterns
    start_marker = "REFERENCES"
    end_marker = "APPENDIX"
    pattern = rf"{re.escape(start_marker)}.*?(?={re.escape(end_marker)}|$)"
    cleaned_text = re.sub(pattern, "", combined_text, flags=re.DOTALL)

    pattern_brackets = r"\[([^\[\]]{1,30},?)+\]"
    cleaned_text = re.sub(pattern_brackets, "", cleaned_text)

    return cleaned_text


# Load documents and setup LLM
def load_documents(pdf_files, prompt_text):
    global vector_store, qa_chain

    # Load and preprocess PDF documents
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file.name)
        documents.extend(loader.load())

    cleaned_text = preprocess_text(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(cleaned_text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_texts(texts, embeddings)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1024)

    # Set up prompt and QA chain
    retrieval_qa_chat_prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=prompt_text,
        max_tokens=4096,
    )
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    return "Documents loaded and model ready for questions."


# Answer question
def answer_question(question):
    if qa_chain is None:
        return "Please load documents first."
    output = qa_chain.invoke({"input": question})
    answer = output["answer"]
    documents_used = output["context"]

    retrieved_content = []
    for document in documents_used:
        text = "..." + document.page_content.replace('\n', ' // ') + "..."
        retrieved_content.append(f"Content:\n{text}\n\n")

    context = ''.join(retrieved_content)
    return answer, context


# Clear inputs
def clear_inputs():
    return "", "", ""


# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("# PDF-based Question Answering with RAG")

    # Step 1: API Key
    gr.Markdown("### Step 1: Enter API Key")
    api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
    set_api_key_button = gr.Button("Set API Key")
    api_key_status = gr.Textbox(label="API Key Status", interactive=False)

    # Step 2: Prompt Input
    gr.Markdown("### Step 2: Set Prompt (Modify if desired, must include {context} and {input})")
    prompt_input = gr.Textbox(label="Custom Prompt", value=default_prompt, lines=10)
    validate_prompt_button = gr.Button("Validate Prompt")
    prompt_status = gr.Textbox(label="Prompt Validation Status", interactive=False)

    # Step 3: Document Upload
    gr.Markdown("### Step 3: Upload PDF Documents")
    pdf_input = gr.Files(label="Upload PDF Documents", file_types=[".pdf"])

    # Step 4: Load Everything
    gr.Markdown("### Step 4: Load Documents and Model")
    load_docs_button = gr.Button("Load Everything")
    load_docs_status = gr.Textbox(label="Load Status", interactive=False)

    # Step 5: Question and Answer
    gr.Markdown("### Step 5: Ask a Question")
    question_input = gr.Textbox(label="Enter your question here")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    context_output = gr.Textbox(label="Context", interactive=False)
    answer_button = gr.Button("Answer Question")

    # Clear button
    clear_button = gr.Button("Clear Question/Answer")

    # Button functionalities
    set_api_key_button.click(set_api_key, inputs=api_key_input, outputs=api_key_status)
    validate_prompt_button.click(
        fn=get_validate_prompt_message,
        inputs=prompt_input,
        outputs=prompt_status
    )
    load_docs_button.click(load_documents, inputs=[pdf_input, prompt_input], outputs=load_docs_status)
    answer_button.click(answer_question, inputs=question_input, outputs=[answer_output, context_output])
    clear_button.click(clear_inputs, outputs=[question_input, answer_output, context_output])

interface.launch(share=True)
