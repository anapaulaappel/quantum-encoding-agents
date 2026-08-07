# When AI Agents Meet Quantum Computing: Why Choosing the Right Embedding Shouldn't Require a PhD

*How LLM-powered agents can make quantum computing accessible to enterprises — with explainability built in.*

---

You've probably heard the pitch: quantum computing will change everything. Drug discovery, financial modeling, supply chain optimization — the list of promised breakthroughs is long. But here's what nobody tells you at the conference keynote: the gap between "quantum computing exists" and "my company can actually use it" is enormous. And a surprising amount of that gap comes down to one deceptively simple question:

**Which embedding do you pick?**

## The Embedding Problem Nobody Talks About

If you've worked with AI systems — RAG pipelines, vector databases, semantic search — you already know embeddings. They're the numerical representations that let machines understand meaning. Pick `all-MiniLM-L6-v2` for speed. Pick `text-embedding-3-large` for accuracy. Pick wrong, and your entire pipeline produces garbage with confidence.

Now take that same problem and drop it into the quantum computing world, and things get significantly harder.

In quantum computing, "embeddings" take a different form. You're mapping classical data into quantum states — a process called *quantum feature mapping* or *data encoding*. And your choices multiply fast:

- **Angle encoding** — maps each feature to a rotation angle on a qubit
- **Amplitude encoding** — encodes data in the amplitudes of a quantum state
- **IQP encoding** — uses entangling gates for higher-order feature interactions
- **Kernel-based encoding** — maps data into a quantum Hilbert space for classification

Each choice affects circuit depth, qubit count, noise sensitivity, and ultimately whether your quantum algorithm actually outperforms its classical counterpart. The wrong encoding strategy doesn't just degrade performance — it can make quantum computing *slower* than a laptop running scikit-learn.

And here's the real problem: **there's no universal rule for which encoding to pick.** It depends on your data distribution, your hardware (IBM Eagle? IBM Heron? A simulator?), your noise profile, your circuit budget. The decision space is combinatorial, and most enterprise teams don't have quantum physicists on staff.

## What If You Could Just Ask?

This is where the idea gets interesting. What if, instead of needing to understand quantum Hilbert spaces, an engineer could type:

> "I have a dataset of 10,000 financial transactions with 15 features. I want to classify fraud. What quantum circuit and encoding should I use, given I have access to a 127-qubit IBM Eagle processor?"

And an AI agent would:

1. Analyze the dataset dimensionality
2. Evaluate encoding strategies against the hardware constraints
3. Build and transpile the quantum circuit
4. Run a simulation to estimate performance
5. Explain every decision it made — in plain language

That's the core idea behind combining **Llama Stack agents** with **IBM's Qiskit**.

## The Architecture: Agents All the Way Down

The project builds on [Llama Stack](https://github.com/meta-llama/llama-stack), Meta's open framework for building AI agent applications. If you've used LangChain or CrewAI, the concept is familiar: an LLM that can call tools, maintain conversation context, and chain actions together. But Llama Stack takes a more opinionated, infrastructure-first approach — it's designed to run as a server, with proper APIs for inference, tool runtime, safety, evaluation, and telemetry.

The architecture looks like this:

```
User (natural language)
       |
       v
 Llama Stack Agent (LLM + Tool-Use + ReAct reasoning)
       |
       +---> Qiskit Tools (circuit creation, transpilation, simulation)
       +---> MCP Servers (OpenShift, Slack, Ansible, GitHub)
       +---> RAG / Vector DB (Milvus — documentation retrieval)
       +---> Web Search (Tavily — latest research, hardware specs)
       |
       v
 Results + Explanation
```

The agent doesn't just execute. It *reasons*. Using the ReAct (Reasoning + Acting) framework, it thinks step by step:

1. "The user wants fraud classification. This is a supervised learning task."
2. "With 15 features, angle encoding would need 15 qubits. That fits within the 127-qubit limit."
3. "But amplitude encoding could compress this into log2(15) ~ 4 qubits, leaving room for error correction."
4. "Let me build both circuits and compare transpilation depth on the Eagle topology."
5. "Amplitude encoding gives a 40% shorter circuit. Recommending this approach."

Every step is visible. Every decision is traceable. That's the explainability part — and for regulated industries like finance and healthcare, it's not optional.

## Why This Matters for Enterprises

Quantum computing has a talent bottleneck. According to McKinsey, fewer than 1,000 people worldwide have deep expertise in quantum algorithms. Meanwhile, IBM has made quantum hardware commercially available, Qiskit has over 600,000 users, and companies are spending real money on quantum R&D.

The disconnect is clear: **the hardware is ready before the workforce is.**

AI agents can bridge this gap in three concrete ways:

### 1. Democratizing Access

An engineer who knows Python and understands their business problem — but not quantum mechanics — can interact with quantum computing through natural language. The agent handles the translation from intent to circuit.

### 2. Accelerating Experimentation

Manually coding, transpiling, and benchmarking quantum circuits is slow. An agent can iterate through encoding strategies, gate decompositions, and optimization passes in minutes. What takes a quantum engineer a week of exploration, an agent can survey in an afternoon.

### 3. Providing Audit Trails

When the agent explains *why* it chose amplitude encoding over angle encoding, or *why* it applied a specific transpilation pass, that reasoning becomes part of the record. For enterprises subject to regulatory scrutiny, this is critical. It's not a black box producing quantum circuits — it's a documented decision chain.

## The Demo Stack: It Runs Locally

One thing that makes this project practical rather than theoretical: it actually runs on your laptop. No GPU cluster required for development. The setup is:

- **Ollama** runs the LLaMA model locally
- **Llama Stack Server** runs in a Podman container (or pure Python)
- **Notebooks and CLI scripts** connect as clients

```bash
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
make setup_local
```

The existing demos (from the [llama-stack-demos](https://github.com/opendatahub-io/llama-stack-demos) project by Red Hat's OpenDataHub team) include a progressive learning path:

- **Level 0-1**: Basic Llama Stack + RAG with vector databases
- **Level 2-3**: Agents with web search, prompt chaining, ReAct reasoning
- **Level 4**: Combining RAG with agentic workflows
- **Level 5-6**: MCP tool integration (Kubernetes operations, Slack notifications)

The Qiskit extension adds a new layer: quantum computing as just another tool the agent can call.

## The Embedding Connection: Full Circle

Here's where it comes together. Remember the embedding problem?

In a traditional RAG pipeline, you pick an embedding model (say, `all-MiniLM-L6-v2` with 384 dimensions), chunk your documents, store vectors in Milvus, and hope your retrieval quality is good enough. In the quantum world, you pick a data encoding strategy, build your feature map circuit, and hope your quantum kernel is expressive enough.

Both are fundamentally the same problem: **how do you represent your data in a space where computation happens?**

An agent that understands both domains can make these decisions holistically:

- "For your classical RAG pipeline, use `all-MiniLM-L6-v2` — your corpus is small and latency matters."
- "For the quantum classification subtask, use amplitude encoding with 4 qubits — your feature space is compact and the Eagle backend has favorable connectivity for this circuit topology."
- "Here's why: [detailed reasoning chain]."

The agent becomes the connective tissue between classical AI infrastructure and quantum computing — speaking both languages fluently and explaining itself to humans who only need to speak one.

## What's Next

The Qiskit agent layer is under active development. The planned roadmap includes:

- Natural language circuit creation and optimization
- Automatic transpilation for specific IBM Quantum backends
- Simulation and execution on real quantum hardware
- Result visualization with histograms and state vectors
- A REST API for integrating quantum capabilities into existing applications

The infrastructure pieces — Llama Stack server, MCP tool protocol, RAG pipeline, observability with Grafana and OpenTelemetry — are already production-tested on OpenShift through the upstream demos.

## The Bigger Picture

We're at an inflection point. LLMs are good enough to understand quantum computing concepts. Quantum hardware is accessible enough through cloud APIs. And agent frameworks like Llama Stack are mature enough to orchestrate complex multi-step workflows with tool calling.

The missing piece was never the technology — it was the interface. An interface that lets a financial analyst say "run a quantum portfolio optimization" without knowing what a Hadamard gate is. An interface that explains its decisions so a compliance officer can audit them. An interface that picks the right embedding — classical or quantum — because a human shouldn't have to.

That's what this project is building. And all it takes to get started is `make setup_local`.

---

*The project is open source and builds on [llama-stack-demos](https://github.com/opendatahub-io/llama-stack-demos) by Red Hat/OpenDataHub and [Qiskit](https://github.com/Qiskit/qiskit) by IBM. Contributions welcome.*
