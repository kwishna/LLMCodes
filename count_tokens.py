import tiktoken

tiktoken_embedding = tiktoken.get_encoding('gpt2')
# tiktoken_embedding = tiktoken.get_encoding('gpt-3.5-turbo')

encoded = tiktoken_embedding.encode("Hello, world!")

decoded = tiktoken_embedding.decode(encoded)