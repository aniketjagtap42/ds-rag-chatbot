from retrieval.vectorstore import load_vectorstore, get_retriever
from generation.llm_client import build_rag_chain

print('Loading vector store...')
db = load_vectorstore()
retriever = get_retriever(db)
chain = build_rag_chain(retriever)

print('Running test question...')
answer = chain.invoke('What is list comprehension?')

print('\n--- ANSWER ---')
print(answer)
