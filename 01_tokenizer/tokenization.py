import tiktoken

enc_model = tiktoken.encoding_for_model("gpt-4o")
enc_model2 = tiktoken.encoding_for_model("code-davinci-002")
text = "My name is Sumit Pareek"

token = enc_model.encode(text)
token2 = enc_model2.encode(text)

print("Tokens: ", token)
print("Tokens2: ", token2)

enc_tokens = [5444, 1308, 382, 34138, 278, 62462, 886]
enc_tokens2 = [3666, 1438, 318, 5060, 270, 350, 533, 988]

decoded = enc_model.decode(enc_tokens)
decoded2 = enc_model2.decode(enc_tokens2)

print("Decoded: ", decoded)
print("Decoded2: ", decoded2)