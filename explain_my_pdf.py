from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredPDFLoader

# Prerequisites
# We are using OpenAI, so export your OpenAI API key with export OPENAI_API_KEY="yourkey"

# 1. Load PDF file using the UnstructuedPDFLoader
# After trial and error, this worked phenomenally better than other loaders, like PyPDF2 or PyMULoader.
# Despite the name, it tends to work on heavily formatted PDF documents well.
loader = UnstructuredPDFLoader("./data/OWASP-Top-10-for-LLMs-2023-v09.pdf")

pages = loader.load()

# 2. Chunk the PDF into sections. RecursiveCharacterTextSplitter is the default text splitter for most Langchain use cases.
#    Play around with chunk size and chunk overlap if you're getting unsatisfactory results recalling information from your model.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len)
sections = text_splitter.split_documents(pages)

# 3. Create a vector store index. There's many ways of doing this, one of the most common is FAISS (Facebook AI Similarity Search)
#    We also pass in the OpenAI Embeddings Object which will make an API call to OpenAI's Embeddings API to generate vectors for your document.
faiss_index = FAISS.from_documents(sections, OpenAIEmbeddings())

# IMPORTANT: Once you've found a good vectorstore index that works well for you, store it in something like Pinecone or Chroma!
#            This will save a ton of OpenAI compute/cost, API latency, and local compute time and power.

# 4. Define your retriever
#    Retrievers use KNN similarity search under the hood!
retriever = faiss_index.as_retriever()

# 5. Choose your model. We're using a fairly high temperature here, tweaking lower will produce responses that are less prone to hallucination with less creativity.
#    You currently need to spend $1 with OpenAI to get access to the GPT-4 API, I'm sure this will change soon though.
turbo_llm = ChatOpenAI(temperature=1,model_name='gpt-3.5-turbo')

# 6. The prompt template is a staple of successful prompt outputs. Here we give instructions relevant to our use case, give model instructions to indicate if it
#    doesn't know something, and the last instruction is the output format. Very basic for demonstration purposes.
prompt_template = """You are an application security engineer. Answer the following question.

Context: {context}

Question: {question}
Return two paragraphs per item.
The output does not need to be serious, it is partially used for comedic purposes.
"""

PROMPT = PromptTemplate(
template=prompt_template, input_variables=["context", "question"])

# Chain arguments
chain_type_kwargs = {"prompt": PROMPT}

# 7. We perform a Retrieval QA on the most basic chain - "stuff" - for demonstration. We explicitly pass in the retriever object from the vectorstore index.abs
#    Chains can get a lot more complex. Stuff is best suited for applications that are small and only a few docs are passed in for most calls per Langchain docs.
qa = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

# 8. The fun part, our question. Change this to user input to play around with your model more.
question = "Explain each OWASP LLM vulnerability using dry, sarcastic language while also making up a story of an acquaintenance being exploited by it."
llm_response = qa(question)

# 9. The actual best part, see all your hard work come together!
print(llm_response["result"])

