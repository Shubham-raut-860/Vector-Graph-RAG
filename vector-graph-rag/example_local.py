import os
# Set a mock API key if not present (this will still fail if it tries to actually make an HTTP call)
os.environ.setdefault("OPENAI_API_KEY", "sk-mock-key")

from vector_graph_rag import VectorGraphRAG

# Initialize VectorGraphRAG using local Milvus Lite
# By default, milvus_uri is "./milvus.db" when not provided, but we can specify it
rag = VectorGraphRAG(milvus_uri="./test_graph.db")

print("Initializing Graph with predetermined triplets (skipping LLM extraction)...")

# We use add_documents_with_triplets to bypass LLM extraction
rag.add_documents_with_triplets([
    {
        "passage": "Albert Einstein developed the theory of relativity at Princeton.",
        "triplets": [
            ["Albert Einstein", "developed", "theory of relativity"],
            ["Albert Einstein", "worked at", "Princeton"],
        ],
    },
    {
        "passage": "The theory of relativity revolutionized physics.",
        "triplets": [
            ["theory of relativity", "revolutionized", "physics"],
        ],
    }
])

print("\nGraph Stats:", rag.get_stats())

print("\nNOTE: Querying requires an active OpenAI API Key to generate embeddings for the search query and answer reranking.")
print("If you have an OpenAI API Key set in your environment, you can run:")
print('result = rag.query("What did Einstein develop?")')
print('print("Answer:", result.answer)')

# Try to run a query if we have a real key, otherwise catch the Auth error
if os.environ.get("OPENAI_API_KEY") != "sk-mock-key":
    try:
        result = rag.query("What did Einstein develop?")
        print("\nAnswer:", result.answer)
        print("Expanded Subgraph Nodes:", len(result.subgraph.entity_ids))
    except Exception as e:
        print("\nQuery failed (likely due to invalid API key):", str(e))
else:
    print("\nSkipping query execution since OPENAI_API_KEY is just a mock key.")
