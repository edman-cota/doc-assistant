import os
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from firecrawl import FirecrawlApp
from langchain.schema import Document

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

def ingest_docs2() -> None:
    urls = ['https://blog.hubspot.com/website/react-js', 
            'https://www.simplilearn.com/tutorials/reactjs-tutorial/what-is-reactjs',
            'https://www.sanity.io/glossary/react-js',
            'https://en.wikipedia.org/wiki/React_(JavaScript_library)',
            'https://codeinstitute.net/global/blog/what-is-react-js/']

    for url in urls:
        app = FirecrawlApp(api_key=os.environ['FIRECRAWL_API_KEY'])

        page_content = app.scrape_url(url=url, params={"onlyMainContent": True})

        print(page_content)
        doc = Document(page_content=str(page_content), metadata={"source": url})

        # Split document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents([doc])

        PineconeVectorStore.from_documents(docs, embeddings, index_name='firecrawl-react-js-index')


if __name__ == '__main__':
    ingest_docs2()
