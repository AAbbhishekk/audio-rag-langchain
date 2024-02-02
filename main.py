from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import streamlit as st
from tempfile import NamedTemporaryFile


load_dotenv()


def create_qa_prompt() -> PromptTemplate:
    """
    Prompt for retrieval QA chain
    """

    template = """\n\nHuman: Use the following pieces of context to answer the question at the end. If answer is not clear, say I DON"T KNOW
{context}
Question: {question}
\n\nAssistant:
Answer:"""

    return PromptTemplate(template=template, input_variables=["context", "question"])


def create_docs(urls_list):
    l = []
    for url in urls_list:
        print(f'Transcribing {url}')
        l.append(AssemblyAIAudioTranscriptLoader(file_path=url).load()[0])
    return l

def make_embedder():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction")

    # return HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )

def make_qa_chain(db):
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },)   
    return RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}),
        return_source_documents=True,
        chain_type_kwargs={
                "prompt": create_qa_prompt(),
            }
    )







# Streamlit UI
def main():
    st.title("Audio Query App")

    # Upload audio file
    uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

    if uploaded_file is not None:
        print('Transcribing files ... (may take several minutes)')
        with NamedTemporaryFile(suffix='.mp3') as temp:
            temp.write(uploaded_file.getvalue())
            temp.seek(0)
            docs = create_docs([temp.name])


        
            print('Splitting documents')
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(docs)

            # modify metadata because some AssemblyAI returned metadata is not in a compatible form for the Chroma db
            for text in texts:
                text.metadata = {"audio_url": text.metadata["audio_url"]}

            # Make vector DB from split texts
            print('Embedding texts...')
            hf = make_embedder()
            db = FAISS.from_documents(texts, hf)
            # db = Chroma.from_documents(texts, hf)

            # Create the chain and start the program
            
            qa_chain = make_qa_chain(db)

            with st.form(key="form"):
                user_input = st.text_input("Ask your question")
                submit_clicked = st.form_submit_button("Submit Question")
            
            if submit_clicked:
                
                result = qa_chain({"query": user_input})
                print(f"Q: {result['query'].strip()}")
                print(f"A: {result['result'].strip()}\n")
                print("SOURCES:")
                for idx, elt in enumerate(result['source_documents']):
                    print(f"    Source {idx}:")
                    print(f"        Filepath: {elt.metadata['audio_url']}")
                    print(f"        Contents: {elt.page_content}")
                print('\n')
            # Process the audio file
            # audio_data, sample_rate = process_audio(uploaded_file)

            # st.audio(audio_data, format='audio/wav', start_time=0)
                
                st.write(result)

                

                # if st.button("Submit Query"):
                #     # Perform some processing on the query (you can customize this part)
                #     result = "Query result: Your query was: '{}'".format(query)
                #     st.success(result)

if __name__ == "__main__":
    main()
