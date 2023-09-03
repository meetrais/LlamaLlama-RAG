from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def SetCustomPrompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def RetrievalQAChain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}), 
                                           return_source_documents=True, chain_type_kwargs={'prompt': prompt})
    return qa_chain

def LoadLLM():
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def LLamaQABot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH,embeddings)
    llm = LoadLLM()
    qaPropmpt = SetCustomPrompt()
    qaBot = RetrievalQAChain(llm,qaPropmpt,db)

    return  qaBot

def GetAnswer(query):
    qaBot = LLamaQABot()
    response = qaBot({'query':query})

    return response

@cl.on_chat_start
async def Start():
    chain=LLamaQABot()
    message = cl.Message(content="Starting Medical Chat Bot...")
    await message.send()
    message.content = "Hi, Welcome to Medical Chat Bot. How may I help you?"
    await message.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    langchainCallBack = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])

    langchainCallBack.answer_reached=True
    response = await chain.acall(message, callbacks=[langchainCallBack])
    answer = response["result"]
    sources = response["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()









    