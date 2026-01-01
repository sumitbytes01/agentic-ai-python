LLM:  
    chatgpt(OpenAI), gemini(Google), claude(Anthropic) LLaMA(META) TITAN(Amazon AWS)
    large language model
    deep learning model
    designed to interact with human language
    trained on massive amount of data
    trained through self supervised learning
    billons of parameters
    transformer architecture
    LLMs can answer questions,
         summarize text, 
         write essays or code, 
         translate languages, 
         and even explain concepts.
    can talk in human language


How LLM works:
GPT: Generative(in Nature) Pretrainined(generate content based on pretrained data) Transformer
How goodle search works: indexing
Input token -> LLM -> Output token

Transformers: generative in nature and work on pretrained models
    Attention is all you need
    input language -> Google Translator -> output language
    input token -> GPT(Transformer) -> Predict the next token
    Hi There -> Transformer -> I
    Hi There I -> Transformer -> am
    Hi There I am-> Transformer -> good.
    Hi There I am good.-> Transformer -> <END>
    Taking input token and producing next set of tokens
LLM models require GPU.

Tokenization: the process of converting user input into numerical representations (tokens) that 
can be understood by LLMs.
A = 1
B = 2
C = 3
D = 4
E = 5
computers are more comfortable with maths.
ABC -> Transformer -> D
123 -> Transformer -> 4
1234 -> Transformer -> 5
Detokenization -> return to user ABCDE
Tokens: Transformers map the human written text to numbers known as tokens
Token generation system is different for all models

toktokenizer

create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    pip install tiktoken


Attention is all you need: Whitepaper
 Input -> (Tokenization -> Input Embedding(Vector Embeddings)) -> Positional Encoding -> Miltihead Attention
    Linear -> Softmax
 Output -> Output Embeddings -> Positional Encoding -> Masked Multi-Head Attention

 Input(Hey there how are you)

 ML - Machine Learning -> develop foundational models - research work
 Application developers -> solving business needs -> no need to for mathematics formulas

Input Embedding - Vector Embeddings - 
    gives semantic(connected with the meaning of word and sentences) meaning to tokens.
    - numeric representation of data points, including text, 
        images, and other data types, that captures their meaning and relationship.
    The dog ate cat
    The mobile 
    Paris, Effiel Tower, India, India Gate

Positional Encoding
    Dog ate cat 
    cat ate dog
        they are going to have same vector embeddings. but the menaing is alltogether different.
        vector encoding cannot differentiate between them
    Dog Ate Cat 
    step1 
        tokenization -> Dog Ate Cat
                        56  74  89
        Vector Embeddings -> 0  1   2
        Positional Encoding -> info about the numbers -> this will say the sentences are different 
            because of position of the vectors
        whole meaning of the sentence will change if the position encoding is not done correctly.

Self-Attention mechanism
    positional enoded embeddings can talk to each other and can actually manipuate and change their meaning
    River bank
    ICICI bank
Multi-head attention mechanism
    takes care of multiple aspects
    dog in the train -> black, breed, sleeping

Feed forward -> neural network, pretict and forward

Linear -> Probability matrix
          Probability of next tokens
          10 tokens with probabilities

Softmax -> taking out the most Probability 
           you can tune up and down the Softmax
    
Tokenizer coding example - tiktoken library
OpenAi connecting and calling api 
Gemini connecting and calling api
pip install google-genai
gemini-OpenAi using open-ai library and gemini free key to call apis and get response

Prompting
    System Prompt Should Be:
        Short
        Stable
        Non-negotiable
    User Prompt Should Be:
        Flexible
        Task-focused
        User-controlled
    Zero-shot-prompting
    few-shot-prompting
    using few-shot
    persona based prompting
types of prompting:
    zero-shot
    few-shotchain-of-thoughts
    automate-chain-of-thoughts
    persona-based

Prompt styles:
    Alpaca prompt - Alpaca-style prompting uses an instruction–input–response format
    ### Instructions: <SYSTEM_PROMPT>\n
    ## Input: <USER_QUERY>
    ### Response:\n

CHATML style prompt:

        {"role": "system" | "user" | "assistant(output your are passing back for the next set of tokens)"},
        {"content": "string"}

        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello there, can you tell me a joke?"}
        
Alpaca style prompt:
    ### Instruction:
        Hello there, can you tell me a joke?

    ### Input:
        SYSTEM_PROMPT: <insert your system prompt here>

    ### Response:

    OR
    If you want to inline the system prompt directly (rather than listing it separately), you can combine them more naturally:
    ### Instruction:
    <insert SYSTEM_PROMPT here>

    ### Input:
        User says: Hello there, can you tell me a joke?

    ### Response:

# INST Prompting - often used by models like LLaMA, Alpaca, Vicuna, Mistral-Instruct, etc
    [INNST] What is the time now? [/INST]
    <|system|>
    <SYSTEM_PROMPT>
    <|user|>
    <USER_MESSAGE>
    <|assistant|>

    <|system|>
    You are a helpful assistant.
    <|user|>
    Hello there, can you tell me a joke?
    <|assistant|>


Aspect	                   Chain of Thought	                    ChatML
Purpose	            Encourage step-by-step reasoning	Structure multi-turn chat messages
Prompt Style	    Natural language + reasoning cues	Special tokens/tags for message roles
Example cue	        "Let's think step by step."	        `<
Use Case	        Problem-solving, math, logic	    Chatbots, structured conversations

OLLAMA₹
    you can also install docker image for olama

    Install docker container for ollama
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    : it will run a container for ollama on a specific port. Now we want to access the Ollama, 
    for that we need the below tool.

    open-webui
    pip install open-webui
    open-webui serve
    http://localhost:8080
    docs: https://github.com/open-webui/open-webui 

    or you can start a container for open-webui as well

Hugging Face
    hugging face is GITHUB of LLM models
    pip install -U "huggingface_hub"
    huggingface-cli login
    pip install transformers

    instead of using local machine, we can use the hugging face to run the models too.

    Models:
    gated models 

    login to hugging face cli using token
    this token would be used for interacting with the hugging face. pulling model etc
    install transformer package

    HF - 
    platform for AI collaboration and innovation
    democratise AI
    transformer library for ease model usage


Agentic AI
    LLM: dumb piece of code sitting on a server taking text as input and giving text as output

    Agent: make llm capable enough to talk to other parts/services(make decisions)

    LLM(Brain) + legs + arms etc = Agent

    LLM + Tools = Agent

    creating an agentic ai use Case:
        creating multiple tools like weather app, some os comands for creating files, folders app etc
        making them available in available_tools 
        creating SYSTEM_PROMPT:
            rules
            output format
            available tools
            examples:
                chain of thoughts:
                    with steps defined
        create class for output format
        create variable to maintainmessage history:
            put in SYSTEM_PROMPT, user USER_QUERY
            on tht basis of user USER_QUERY:
                keep appending raw result to message history as assistant
                    keep checking the step type after parsing raw result and making decision to continue/break or output
                    
RAG
    **Retrieval-Augmented Generation**
    AI framework
    improves the accuracy and relevance of large language model (LLM) responses
    combines information retrieval with text generation
    Instead of relying solely on the model's pre-trained knowledge, RAG fetches relevant information from:
        external sources (such as databases, documents, or APIs)before generating a response.
    This allows the model to provide 
        up-to-date, 
        contextually accurate, and 
        domain-specific answers 
    without needing to retrain the underlying LLM.

    How RAG Works

    Retrieval Step: When a user submits a query, RAG uses an information retrieval system to search for relevant documents or data from an external knowledge base. This is often done using embeddings and vector databases to find semantically similar content.​

    Augmentation Step: The retrieved information is combined with the original query and fed into the LLM.

    Generation Step: The LLM uses both its own knowledge and the retrieved context to generate a more accurate and grounded response.​

    Key Benefits

    Improved Accuracy: Reduces hallucinations and ensures responses are factually correct by referencing authoritative sources.​

    Up-to-Date Information: Allows models to access real-time or domain-specific data, not just static training data.​

    Domain Expertise: Enables LLMs to answer questions in specialized fields (e.g., medicine, law, finance) by leveraging external knowledge bases.​

    Cost-Effective: Avoids the need to retrain or fine-tune the LLM for every new dataset or domain.​

    Common Use Cases

    Question-answering systems and chatbots

    Content creation and summarization

    Conversational agents in enterprise settings

    Educational tools and research assistants​

    RAG is widely adopted by major tech companies and is considered a best practice for building reliable, context-aware generative AI applications.

    https://cdn.hashnode.com/res/hashnode/image/upload/v1724944925051/e525c6cb-6a99-4eec-8b47-3dc827ddff25.png

    ![alt text](image.png)

    indexing phase - provide the data
    retrieval phase - chatting the data

    INDEXING (offline)
    Docs → Chunks → Embeddings → Vector DB

    RETRIEVAL (online)
    Query → Embedding → Vector DB → Context → LLM → Answer

    RAG example using langchain:
        indexing:
            read pdf using pyPDFLoader
                slpit text using langchanin and prepare chunks
                create vector embeddings of the chunks
                store in vector DB - qdrant
        generation:
            load vector embeddings
            connect to vector db - qdrant
            USER_QUERY
            do a similarity search on vectordb
            prepare context
            prepare system prompt with the context
            call LLM and get the results
        



