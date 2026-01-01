import tiktoken

enc_model = tiktoken.encoding_for_model("gpt-4o")
text = "My name is Sumit Pareek"

token = enc_model.encode(text)

print("Tokens: ", token)

enc_tokens = [5444, 1308, 382, 34138, 278, 62462, 1886]

decoded = enc_model.decode(enc_tokens)

print("Decoded: ", decoded)