LangGraph is a framework for building stateful, multi-step AI workflows using graph structures—created as part of the LangChain ecosystem.

🧠 In simple terms
    LangGraph lets you design AI systems like a flowchart with memory, where:
        Each node = a step (LLM call, tool, retrieval, etc.)
        Each edge = how execution moves
    The system can loop, branch, and remember state
    👉 Think of it as:
    “A more powerful way to build agents than simple chains”

🔄 Why LangGraph exists
    Traditional LangChain “chains” are:
        Linear (step 1 → step 2 → step 3)
        But real AI apps need:
            Loops (retry, refine)
            Decisions (if/else)
            State (memory across steps)
    LangGraph solves this using a graph-based execution model.

⚙️ Core concepts
    1. Nodes
        Each node does some work:
            Call an LLM
            Query a vector DB
            Run a tool
    2. Edges
        Define flow:
            Fixed path
            Conditional routing (like if answer is bad → retry)
    3. State
        Shared memory passed between nodes:
            state = {
                    "question": "...",
                    "documents": [...],
                    "answer": "..."
                    }

Example (RAG with loop)
    LangGraph shines in RAG systems:
        Flow:
            User question
            Retrieve documents
            Generate answer
            Evaluate answer
            ❗ If bad → go back to retrieve (loop)
        This kind of loop is hard in plain LangChain, but easy in LangGraph.

🚀 Key features
    Stateful workflows
    Loops & retries
    Conditional branching
    Multi-agent orchestration
    Streaming support

| Feature   | LangChain | LangGraph        |
| --------- | --------- | ---------------- |
| Flow type | Linear    | Graph (flexible) |
| Loops     | ❌ Hard    | ✅ Native        |
| State     | Limited   | ✅ Strong         |
| Agents    | Basic     | Advanced         |

🧩 When to use LangGraph
    Use it if you're building:
        RAG systems with retries
        AI agents with decision-making
        Multi-step workflows
        Async / background pipelines (like your diagram 👆)

⚠️ When NOT to use it
    Avoid if:
        Your flow is simple (just 1–2 steps)
        You don’t need loops or state

🔥 Mental model
    LangChain = pipeline
    LangGraph = workflow engine