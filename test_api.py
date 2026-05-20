import requests

BASE = 'http://localhost:8000'

def test(question, session_id='test'):
    response = requests.post(f'{BASE}/chat', json={
        'question': question,
        'session_id': session_id
    })
    data = response.json()
    print(f'\nQ: {question}')
    print(f'In scope: {data["in_scope"]}')
    print(f'Rewritten: {data["rewritten_query"]}')
    print(f'A: {data["answer"][:300]}')
    print('-' * 60)

print('=== Testing DS Notes Assistant API ===')

print('\n--- Test 1: Basic question ---')
test('What is list comprehension?')

print('\n--- Test 2: Memory test ---')
test('What is a lambda function?', session_id='aniket_session')
test('Give me a code example of that', session_id='aniket_session')

print('\n--- Test 3: Out of scope ---')
test('What is the weather in Pune today?')

print('\n--- Test 4: Deep Learning question ---')
test('What is dropout in deep learning?')

print('\n--- Test 5: RAG question ---')
test('Explain RAG architecture in simple words')

print('\n--- Test 6: Git question ---')
test('What are the 3 stages of Git workflow?')
