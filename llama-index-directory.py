from llama_index import VectorStoreIndex, SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, StorageContext, \
    load_index_from_storage, OpenAIEmbedding
from llama_index import ServiceContext, LLMPredictor, PromptHelper
from llama_index.llms import OpenAI

max_input=4096
tokens=256
chunk_size=600
max_chunk_overlap=20
data_dir='./data'

prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

llm_predictor = LLMPredictor(llm=OpenAI(temperatur=0.8, model_name='text-ada-001', max_tokens=tokens))

reader = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".txt"], recursive=True)
docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

# service_context = ServiceContext.from_defaults(prompt_helper=prompt_helper, embed_model=OpenAIEmbedding(embed_batch_size=10))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
index = VectorStoreIndex.from_documents(
    docs,
    service_context=service_context,
    show_progress=True
)

index.storage_context.persist(persist_dir='./storage')

# rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="./storage")
# index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
response=query_engine.query("Who was the first prime minister of india?")
print(response)