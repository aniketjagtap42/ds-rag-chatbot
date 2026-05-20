from retrieval.vectorstore import load_vectorstore, get_retriever
from generation.llm_client import build_rag_chain
from agents.router_agent import RouterAgent

print('Loading vector store...')
db = load_vectorstore()
retriever = get_retriever(db)
chain = build_rag_chain(retriever)
agent = RouterAgent(chain)

def test(question, session_id='test'):
    print(f'\nQ: {question}')
    result = agent.run(question=question, session_id=session_id)
    print(f'In scope: {result["in_scope"]}')
    print(f'Rewritten: {result["rewritten_query"]}')
    print(f'A: {result["answer"][:300]}')
    print('-' * 60)

print('\n=== TEST 1: Normal question ===')
test('What is a lambda function?')

print('\n=== TEST 2: Follow-up memory test ===')
test('What is a lambda function?', session_id='memory_test')
test('Give me a code example of that', session_id='memory_test')

print('\n=== TEST 3: Out of scope ===')
test('What is the weather in Pune today?')

print('\n=== TEST 4: Another DS question ===')
test('What is the difference between break and continue?')
