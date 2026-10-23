"""UNDERSTAND (static half): a signature library that tags every symbol with AI facets.

A facet is "this function does retrieval / calls an LLM / is a tool / ...". One function can have
several facets (facet splitting). Signals are framework-agnostic signatures over call sites,
constructors, decorators, imports and names, for Python and JS/TS alike.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..analyzer.extract import MODULE
from ..analyzer.facts import CallFact, DefFact, FileFacts
from ..core.schemas import ConfigValue, Evidence

# facet -> graph node type
FACET_TYPE = {
    "llm": "llm", "embedding": "embedding", "vector_db": "vector_db", "retriever": "retriever",
    "reranker": "reranker", "ingest": "dataset", "tool": "tool", "agent": "agent", "memory": "memory",
    "guardrail": "guardrail", "parser": "parser", "external_api": "external_api", "entrypoint": "entrypoint",
    "router": "router", "prompt": "prompt",
}
# modifiers: not components, but they change bottleneck verdicts
MODIFIERS = {"cache", "retry", "structured", "client"}

LLM_CTORS = {
    "OpenAI", "AzureOpenAI", "AsyncOpenAI", "AsyncAzureOpenAI", "Anthropic", "AsyncAnthropic", "AnthropicBedrock",
    "ChatOpenAI", "AzureChatOpenAI", "ChatAnthropic", "ChatOllama", "Ollama", "OllamaLLM", "ChatGoogleGenerativeAI",
    "GoogleGenerativeAI", "GenerativeModel", "ChatBedrock", "ChatBedrockConverse", "BedrockChat", "ChatGroq", "Groq",
    "ChatMistralAI", "MistralClient", "Mistral", "ChatCohere", "Cohere", "CohereClient", "HuggingFaceEndpoint",
    "HuggingFacePipeline", "ChatHuggingFace", "LlamaCpp", "VLLM", "OpenAILike", "ChatDeepSeek", "ChatTogether",
    "ChatFireworks", "ChatVertexAI", "VertexAI", "ChatLiteLLM", "AzureAIChatCompletionsModel", "ChatPerplexity",
    "ChatXAI", "LLM", "Llama", "AutoModelForCausalLM", "VLLMOpenAI", "LlamaCppModel", "CTransformers",
}
LLM_HOSTS = re.compile(r"api\.openai\.com|openai\.azure\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|"
                       r"api\.cohere\.(ai|com)|api\.groq\.com|api\.mistral\.ai|openrouter\.ai|api\.together\.xyz|"
                       r"api\.deepseek\.com|api\.perplexity\.ai|ai-proxy|:11434|/chat/completions|/v1/completions|"
                       r"/api/generate|/api/chat|/v1/messages|:generateContent|:streamGenerateContent")
EMBED_HOSTS = re.compile(r"/v1/embeddings|/api/embed|/embeddings\b|:embedContent")
ROLE_CLASS = [(re.compile(r"(Generator|LLM|ChatModel|Completion)$"), "llm"), (re.compile(r"(Embedder|Embedding|Vectorizer)$"), "embedding"),
              (re.compile(r"(Chunker|Splitter)$"), "ingest"), (re.compile(r"(Reader|Loader|Ingestor)$"), "ingest"),
              (re.compile(r"Retriever$"), "retriever"), (re.compile(r"Reranker$"), "reranker"),
              (re.compile(r"(Guard|Moderator|Filter)$"), "guardrail")]
LLM_CALL_SUFFIXES = (".chat.completions.create", ".chat.completions.parse", ".completions.create", ".messages.create",
                     ".messages.stream", ".responses.create", ".generate_content", ".generate_content_async",
                     ".converse", ".invoke_model", "litellm.completion", "litellm.acompletion", "ollama.chat",
                     "ollama.generate", ".chat.complete", ".chat.stream")
LLM_FUNCS = {"generateText", "streamText", "generateObject", "streamObject", "completion", "acompletion",
             "create_chat_completion", "create_completion"}
LLM_METHODS = {"invoke", "ainvoke", "stream", "astream", "batch", "abatch", "predict", "apredict", "complete",
               "acomplete", "chat", "achat", "stream_chat", "generate", "agenerate", "run", "arun", "kickoff", "call"}
LLM_OWNER = re.compile(r"llm|chat|model|chain|client|gpt|claude|gemini|bot|openai|anthropic|engine|crew|agent|graph|app|runnable|pipeline", re.I)
AGENT_OWNER = re.compile(r"graph|agent|crew|executor|workflow|app$", re.I)

EMBED_CTOR = re.compile(r"Embeddings?$|^TextEmbedding$|^SentenceTransformer$|^HuggingFaceEmbedding$|^OpenAIEmbedding$|^FastEmbedEmbedding$")
EMBED_CALLS = {"embed_query", "embed_documents", "aembed_query", "aembed_documents", "embedQuery", "embedDocuments",
               "embed", "get_text_embedding", "get_query_embedding"}
VECTOR_CTORS = {"QdrantClient", "AsyncQdrantClient", "QdrantVectorStore", "Qdrant", "Chroma", "FAISS", "Pinecone",
                "PineconeVectorStore", "Weaviate", "WeaviateVectorStore", "Milvus", "PGVector", "PGVectorStore",
                "ElasticsearchStore", "LanceDB", "MongoDBAtlasVectorSearch", "MemoryVectorStore", "HNSWLib",
                "SupabaseVectorStore", "VectorStoreIndex", "SimpleVectorStore", "Neo4jVector", "OpenSearchVectorSearch",
                "AzureSearch", "Redis", "DocArrayInMemorySearch", "InMemoryVectorStore", "ChromaVectorStore",
                "PersistentClient", "EphemeralClient", "HttpClient"}
VECTOR_CALLS = {"get_or_create_collection", "create_collection", "add_documents", "addDocuments", "upsert",
                "add_texts", "addVectors", "create_index"}
INDEX_BUILD = {"from_documents", "from_texts", "fromDocuments", "fromTexts", "fromExistingIndex", "from_existing_index"}
SIMILARITY_WORDS = re.compile(r"(?i)cos|simil|score|dist|\bdot\b|rank")
RANK_CALLS = {"sorted", "nlargest", "nsmallest", "argsort", "argpartition", "topk", "top_k", "most_similar"}
RETRIEVE_CALLS = {"as_retriever", "asRetriever", "similarity_search", "similarity_search_with_score",
                  "similarity_search_with_relevance_scores", "similaritySearch", "similaritySearchWithScore",
                  "max_marginal_relevance_search", "maxMarginalRelevanceSearch", "get_relevant_documents",
                  "getRelevantDocuments", "aget_relevant_documents", "query_points", "as_query_engine", "retrieve",
                  "aretrieve", "hybrid_search"}
RETRIEVE_CTORS = {"BM25Retriever", "EnsembleRetriever", "MultiQueryRetriever", "VectorIndexRetriever",
                  "ParentDocumentRetriever", "SelfQueryRetriever", "TFIDFRetriever"}
STORE_OWNER = re.compile(r"store|index|collection|client|db|vector|qdrant|chroma|pinecone|faiss|retriever|kb", re.I)
RERANK_CTORS = {"CrossEncoder", "TextCrossEncoder", "CohereRerank", "FlashrankRerank", "SentenceTransformerRerank",
                "LLMRerank", "ContextualCompressionRetriever", "CrossEncoderReranker", "JinaRerank", "RankGPTRerank"}
SPLITTER = re.compile(r"(Text)?Splitter$|^SentenceSplitter$|^SemanticChunker$|^TokenTextSplitter$")
INGEST_CALLS = {"split_documents", "split_text", "splitDocuments", "splitText", "load_data", "lazy_load"}
LOADER = re.compile(r"(Loader|Reader)$")
PROMPT_CTORS = {"ChatPromptTemplate", "PromptTemplate", "SystemMessagePromptTemplate", "HumanMessagePromptTemplate",
                "FewShotPromptTemplate", "MessagesPlaceholder"}
TOOL_CTORS = {"StructuredTool", "Tool", "DynamicTool", "DynamicStructuredTool", "FunctionTool", "ToolNode", "BaseTool"}
AGENT_CTORS = {"StateGraph", "MessageGraph", "create_react_agent", "createReactAgent", "AgentExecutor",
               "initialize_agent", "create_tool_calling_agent", "create_openai_tools_agent", "create_openai_functions_agent",
               "Crew", "ConversableAgent", "AssistantAgent", "UserProxyAgent", "GroupChat", "ReActAgent",
               "FunctionCallingAgent", "OpenAIAgent", "AgentRunner", "Swarm", "Agent", "Workflow"}
CHAIN_CTORS = {"LLMChain", "RetrievalQA", "ConversationalRetrievalChain", "create_retrieval_chain",
               "create_stuff_documents_chain", "load_qa_chain", "RunnableSequence", "RunnableParallel",
               "RunnableWithMessageHistory", "ConversationChain", "RetrievalQAWithSourcesChain"}
MEMORY_CTORS = {"ConversationBufferMemory", "ConversationBufferWindowMemory", "ConversationSummaryMemory",
                "ConversationSummaryBufferMemory", "ChatMessageHistory", "InMemoryChatMessageHistory",
                "RedisChatMessageHistory", "MemorySaver", "InMemorySaver", "SqliteSaver", "PostgresSaver",
                "BufferMemory", "BufferWindowMemory", "ChatMemoryBuffer", "ConversationTokenBufferMemory", "Memory"}
UNBOUNDED_MEMORY = {"ConversationBufferMemory", "ChatMessageHistory", "InMemoryChatMessageHistory", "BufferMemory"}
GUARD_CALLS = {"moderations.create", "Guard", "LLMRails", "RailsConfig", "AnalyzerEngine", "scan_prompt", "scan_output"}
PARSER_STRUCTURED = {"PydanticOutputParser", "JsonOutputParser", "with_structured_output", "withStructuredOutput",
                     "StructuredOutputParser", "OutputFixingParser", "generateObject", "model_validate_json",
                     "parse_raw", "instructor", "from_openai"}
HTTP = re.compile(r"^(requests|httpx|aiohttp|urllib\.request|axios|session|client)\.(get|post|put|patch|delete|request|urlopen)$|^fetch$|^axios$|^got$")
CACHE_DECOS = re.compile(r"(^|\.)(lru_cache|cache|cached|cache_resource|cache_data|memoize)$")
CACHE_CALLS = {"set_llm_cache", "InMemoryCache", "SQLiteCache", "RedisCache", "Cache", "GPTCache", "RedisSemanticCache"}
RETRY_DECOS = re.compile(r"(^|\.)(retry|on_exception|retry_with_exponential_backoff)$")
ROUTE_DECOS = re.compile(r"\.(get|post|put|patch|delete|route|websocket|api_route)$")
ENTRY_CALLS = {"chat_input", "text_input", "text_area", "ChatInterface", "Interface", "Blocks", "launch", "queue", "run_polling",
               "listen", "on_message", "cl.on_message"}
NAME_FACETS = [
    (re.compile(r"guardrail|moderat|pii|screen_input|sanitiz|safety_check|validate_input|content_filter"), "guardrail"),
    (re.compile(r"rerank"), "reranker"),
    (re.compile(r"retriev|search_docs|get_context|fetch_context|similar_docs"), "retriever"),
    (re.compile(r"^(route|router|classify|intent|dispatch|select_tool|choose_tool|plan)"), "router"),
    (re.compile(r"memory|history|remember|recall"), "memory"),
    (re.compile(r"^parse|_parse|parser"), "parser"),
    (re.compile(r"^embed|embedding"), "embedding"),
]
PROMPT_NAME_FACETS = [
    (re.compile(r"ROUTER|ROUTING|CLASSIF|INTENT"), "router"),
    (re.compile(r"GUARD|MODERAT|SAFETY"), "guardrail"),
    (re.compile(r"REWRITE|CONDENSE|REPHRASE"), "llm"),
    (re.compile(r"REVIEW|GRADE|JUDGE|CRITIC"), "llm"),
    (re.compile(r"INJECTION|BLOCKLIST|BLOCK_LIST|BANNED|DENYLIST|JAILBREAK|TOPIC_WORDS|ALLOWED_TOPICS"), "guardrail"),
    (re.compile(r"MEMORY|HISTORY|MAX_TURNS|SESSION"), "memory"),
]
NODE_TYPE_NAMES = {"entrypoint", "agent", "router", "llm", "prompt", "retriever", "vector_db", "embedding", "reranker",
                   "tool", "memory", "guardrail", "external_api", "parser", "dataset"}


@dataclass
class Signal:
    facet: str
    weight: float
    line: int
    snippet: str
    fact: str
    provider: str | None = None
    params: dict[str, ConfigValue] = field(default_factory=dict)
    via: str | None = None        # symbol key where the bound object was constructed (binding signals)

    @property
    def weak(self) -> bool:
        """Hints (names, declarations, prompt names) as opposed to direct evidence (calls, constructors, decorators)."""
        return self.fact.startswith(WEAK_PREFIXES)


WEAK_PREFIXES = ("name suggests", "declared as", "uses prompt/const", "works with conversation state")
HISTORY_NAMES = {"history", "chat_history", "messages", "turns", "conversation", "memory", "session_history"}


@dataclass
class SymbolInfo:
    key: str                 # "path::qualname"
    path: str
    d: DefFact
    signals: list[Signal] = field(default_factory=list)

    def weight(self, facet: str) -> float:
        return sum(s.weight for s in self.signals if s.facet == facet)

    def facets(self, threshold: float = 0.5, limit: int = 6) -> list[str]:
        scores: dict[str, float] = {}
        for s in self.signals:
            if s.facet in FACET_TYPE:
                scores[s.facet] = scores.get(s.facet, 0) + s.weight
        ranked = sorted((f for f, w in scores.items() if w >= threshold), key=lambda f: -scores[f])
        return ranked[:limit]

    def has(self, facet: str) -> bool:
        return any(s.facet == facet for s in self.signals)

    def evidence(self, facet: str, limit: int = 3) -> list[Evidence]:
        sigs = sorted((s for s in self.signals if s.facet == facet), key=lambda s: -s.weight)[:limit]
        return [Evidence(file=self.path, line=s.line, snippet=s.snippet, fact=s.fact) for s in sigs]

    def params(self, facet: str | None = None) -> dict[str, ConfigValue]:
        out: dict[str, ConfigValue] = {}
        for s in self.signals:
            if facet is None or s.facet == facet:
                out.update(s.params)
        return out

    def provider(self, facet: str) -> str | None:
        return next((s.provider for s in self.signals if s.facet == facet and s.provider), None)


def _parts(call: CallFact) -> list[str]:
    return [p for p in call.callee.replace("()", "").replace("[]", "").split(".") if p]


def _owner(call: CallFact) -> str:
    parts = _parts(call)
    return parts[-2] if len(parts) > 1 else ""


def _provider_of(name: str, imports: set[str], dial: bool) -> str | None:
    n = name.lower()
    if "azure" in n:
        return "dial" if dial else "azure-openai"
    for key, prov in [("anthropic", "anthropic"), ("claude", "anthropic"), ("openai", "openai"), ("gpt", "openai"),
                      ("ollama", "ollama"), ("google", "google"), ("gemini", "google"), ("vertex", "google"),
                      ("bedrock", "bedrock"), ("groq", "groq"), ("mistral", "mistral"), ("cohere", "cohere"),
                      ("huggingface", "huggingface"), ("litellm", "litellm"), ("deepseek", "deepseek")]:
        if key in n:
            return prov
    for mod in imports:
        if mod in ("openai", "anthropic", "litellm", "ollama", "cohere", "groq", "mistralai"):
            return "dial" if (mod == "openai" and dial) else mod
    return None


def _scalar_params(call: CallFact) -> dict[str, ConfigValue]:
    return {k: v for k, v in call.kwargs.items() if isinstance(v, (int, float, bool)) or v is None or
            (isinstance(v, str) and len(v) < 80)}


def tag_call(call: CallFact, imports: set[str], dial: bool) -> list[Signal]:
    """Signals contributed by one call site."""
    parts = _parts(call)
    if not parts:
        return []
    last, owner, name = parts[-1], _owner(call).lower(), call.callee
    sig: list[Signal] = []

    def add(facet: str, weight: float, fact: str, provider: str | None = None, params: bool = True) -> None:
        sig.append(Signal(facet, weight, call.line, call.snippet, fact, provider,
                          _scalar_params(call) if params else {}))

    if name.endswith(LLM_CALL_SUFFIXES) or last in LLM_FUNCS:
        add("llm", 1.0, f"calls {name}", _provider_of(name, imports, dial))
        if "tools" in call.kwargs or "functions" in call.kwargs or "tool_choice" in call.kwargs:
            if call.in_loop or "maxSteps" in call.kwargs or "max_steps" in call.kwargs:
                add("agent", 1.0, "tool-calling loop around the LLM (agent)")
            else:
                add("agent", 0.4, "LLM call with tool definitions (function calling)")
        if "response_format" in call.kwargs or name.endswith(".parse"):
            add("structured", 1.0, "structured output requested")
    elif last in LLM_CTORS or (len(parts) == 1 and last in {"openai", "anthropic", "azure", "google"} and call.str_args):
        add("client", 1.0, f"creates LLM client {last}", _provider_of(last, imports, dial))
    elif last in LLM_METHODS and owner and LLM_OWNER.search(owner):
        if AGENT_OWNER.search(owner):
            add("agent", 0.6, f"runs {name}")
        else:
            add("llm", 0.8, f"invokes {name}")
    if any(p in AGENT_CTORS for p in parts):
        add("agent", 1.0, f"builds agent/graph via {last}", "langgraph" if "Graph" in name else None, params=False)
    if any(p in CHAIN_CTORS for p in parts):
        add("agent", 0.8, f"builds chain {last}", params=False)
        if "Retrieval" in name or "retriever" in call.kwargs:
            add("retriever", 0.6, f"chain retrieves via {last}")
    if EMBED_CTOR.search(last) or last in EMBED_CALLS or name.endswith("embeddings.create"):
        add("embedding", 1.0, f"embeddings via {name}")
    if last in VECTOR_CTORS and (last not in {"PersistentClient", "EphemeralClient", "HttpClient", "Redis"} or "chromadb" in imports):
        add("vector_db", 1.0, f"vector store {last}", last.lower().replace("vectorstore", ""))
    if last in VECTOR_CALLS:
        add("vector_db", 0.8, f"writes vectors via {name}")
    if last in INDEX_BUILD:
        add("vector_db", 0.9, f"builds index via {name}")
        add("ingest", 0.6, f"indexes documents via {name}")
    if last in RETRIEVE_CALLS or last in RETRIEVE_CTORS or (last in {"search", "query"} and STORE_OWNER.search(owner)):
        add("retriever", 1.0, f"retrieves via {name}")
    if last in RERANK_CTORS or (last == "rerank" and owner):
        add("reranker", 1.0, f"reranks via {name}")
    if SPLITTER.search(last) or last in INGEST_CALLS or (LOADER.search(last) and last[0].isupper()):
        add("ingest", 0.9, f"ingests documents via {last}")
    if last in PROMPT_CTORS or last in {"from_template", "fromTemplate", "from_messages", "fromMessages"} or name == "hub.pull":
        add("prompt", 0.7, f"prompt template via {name}", params=False)
    if last in TOOL_CTORS or (last == "tool" and len(parts) == 1):
        add("tool", 0.8, f"defines tool via {last}", params=False)
    if last in MEMORY_CTORS or last in {"add_message", "add_messages", "add_user_message", "add_ai_message"}:
        add("memory", 1.0, f"conversation memory via {last}")
        if last in UNBOUNDED_MEMORY:
            add("memory_unbounded", 1.0, f"{last} keeps the full history")
    if "checkpointer" in call.kwargs:
        add("memory", 0.45, "graph compiled with a checkpointer")
    if any(g in name for g in GUARD_CALLS):
        add("guardrail", 1.0, f"guardrail via {name}")
    if name in ("json.loads", "JSON.parse"):
        add("json_parse", 1.0, f"parses JSON via {name}", params=False)
    if any(p in PARSER_STRUCTURED for p in parts):
        add("structured", 1.0, f"structured output via {last}", params=False)
        add("parser", 0.5, f"output parser {last}", params=False)
    is_http = HTTP.match(name) or (last in {"get", "post", "stream"} and owner in {"requests", "httpx", "axios", "session", "client"})
    if is_http:
        url = " ".join([*call.str_args, *[str(v) for v in call.kwargs.values()], call.snippet])
        if LLM_HOSTS.search(url):
            add("llm", 1.0, f"raw HTTP call to an LLM API via {name}")
        elif EMBED_HOSTS.search(url):
            add("embedding", 1.0, f"raw HTTP call to an embeddings API via {name}")
        else:
            add("external_api", 0.9, f"HTTP call {name}")
    if last in CACHE_CALLS:
        add("cache", 1.0, f"cache via {last}", params=False)
    if "max_retries" in call.kwargs or "maxRetries" in call.kwargs:
        add("retry", 0.8, "client retries configured", params=False)
    if last in {"get", "post", "put", "patch", "delete", "all"} and owner in {"app", "router", "server", "api"} \
            and call.str_args and call.str_args[0].startswith("/"):
        add("entrypoint", 1.0, f"web route @{owner}.{last} {call.str_args[0]}", params=False)
    elif last in ENTRY_CALLS and owner in {"st", "gr", "bot", "cl", "client", ""}:
        add("entrypoint", 0.9, f"user interface via {name}", params=False)
    return sig


def tag_symbols(files: list[FileFacts], dial: bool) -> dict[str, SymbolInfo]:
    symbols: dict[str, SymbolInfo] = {}
    for f in files:
        if f.error:
            continue
        imports = {i.module.split(".")[0].split("/")[0].lstrip("@") for i in f.imports}
        for d in f.defs:
            if d.kind != "function":
                continue
            info = SymbolInfo(key=f"{f.path}::{d.qualname}", path=f.path, d=d)
            for call in d.calls:
                info.signals += tag_call(call, imports, dial)
            # hand-rolled vector search: embeds the query and ranks candidates by similarity (no vector-DB library)
            ranks = [c for c in d.calls if c.callee.replace("()", "").split(".")[-1] in RANK_CALLS
                     and SIMILARITY_WORDS.search(c.snippet)]
            if ranks and any(s.facet == "embedding" and s.weight >= 1.0 for s in info.signals):
                info.signals.append(Signal("retriever", 0.8, ranks[0].line, ranks[0].snippet,
                                           f"ranks embedded candidates via {ranks[0].callee} (hand-rolled vector search)"))
            for deco in d.decorators:
                if deco in ("tool", "function_tool") or deco.endswith((".tool", ".function_tool")):
                    info.signals.append(Signal("tool", 1.0, d.line, d.snippet, f"@{deco} decorator"))
                if ROUTE_DECOS.search(deco) and deco.split(".")[0] in {"app", "router", "api", "bp", "blueprint"}:
                    info.signals.append(Signal("entrypoint", 1.0, d.line, d.snippet, f"web route @{deco}"))
                if CACHE_DECOS.search(deco):
                    info.signals.append(Signal("cache", 1.0, d.line, d.snippet, f"@{deco}"))
                if RETRY_DECOS.search(deco):
                    info.signals.append(Signal("retry", 1.0, d.line, d.snippet, f"@{deco}"))
            owner_cls = d.qualname.split(".")[0] if "." in d.qualname else None
            if owner_cls and not d.name.startswith("_"):
                for pattern, facet in ROLE_CLASS:
                    if pattern.search(owner_cls):
                        info.signals.append(Signal(facet, 0.35, d.line, d.snippet, f"name suggests {facet} (class {owner_cls})"))
                        # an HTTP call in a model-calling method of an LLM/embedding plugin class IS the model call
                        if facet in ("llm", "embedding") and re.search(r"generat|chat|complet|stream|invoke|predict|answer|embed|vector", d.name, re.I):
                            for s in list(info.signals):
                                if s.facet == "external_api":
                                    info.signals.remove(s)
                                    info.signals.append(Signal(facet, 1.0, s.line, s.snippet,
                                                               f"{s.fact} inside {owner_cls}.{d.name} (model API)", s.provider, s.params))
                        break
            declared = next((a for a in d.decorator_str_args if a in NODE_TYPE_NAMES), None)
            if declared:  # one weak hint, never decisive on its own
                facet = next((k for k, v in FACET_TYPE.items() if v == declared), declared)
                info.signals.append(Signal(facet, 0.4, d.line, d.snippet, f'declared as "{declared}" by a decorator'))
            # names of referenced prompt constants carry intent (ROUTER_PROMPT, CLASSIFY_SYSTEM, GUARD_PROMPT, ...)
            prompt_names = {p.name for p in f.prompts} | {c.name for c in f.consts}
            for used in d.names_used:
                if used in prompt_names or used.isupper():
                    for pattern, facet in PROMPT_NAME_FACETS:
                        if pattern.search(used):
                            info.signals.append(Signal(facet, 0.4, d.line, d.snippet, f"uses prompt/const {used}"))
                            break
            if any(h in n.lower() for n in d.names_used + d.args for h in HISTORY_NAMES):
                info.signals.append(Signal("memory", 0.3, d.line, d.snippet, "works with conversation state"))
            lname = d.name.lower()
            for pattern, facet in NAME_FACETS:
                if d.name != MODULE and pattern.search(lname):
                    weight = 0.5 if facet in ("guardrail", "memory", "reranker", "retriever") else 0.35
                    info.signals.append(Signal(facet, weight, d.line, d.snippet, f"name suggests {facet}"))
                    break
            if d.name in ("POST", "GET") and re.search(r"(^|/)route\.(ts|js)$", f.path):
                info.signals.append(Signal("entrypoint", 1.0, d.line, d.snippet, "Next.js route handler"))
            if d.name == "lambda_handler" or d.name == "handler" and "aws" in f.path:
                info.signals.append(Signal("entrypoint", 0.8, d.line, d.snippet, "serverless handler"))
            if d.name == MODULE and f.has_main_guard:
                info.signals.append(Signal("entrypoint", 0.5, 1, "", "script entry (__main__)"))
            # parser: json parsing in a symbol that is itself LLM-facing or named like a parser
            direct_llm = any(s.facet == "llm" and not s.weak for s in info.signals)
            named_parser = bool(re.search(r"(^|_)parse", d.name, re.I))
            if info.has("json_parse") and (direct_llm or named_parser
                                           or any(s.facet == "parser" and not s.weak for s in info.signals)):
                line = next(s.line for s in info.signals if s.facet == "json_parse")
                info.signals.append(Signal("parser", 0.6, line, "", "parses LLM output as JSON"))
            symbols[info.key] = info
    _resolve_bindings(files, symbols, dial)
    _registry_tools(files, symbols)
    return symbols


def _resolve_bindings(files: list[FileFacts], symbols: dict[str, SymbolInfo], dial: bool) -> None:
    """`llm = ChatOpenAI(temperature=0.9)` somewhere + `llm.invoke(...)` elsewhere: the user of the client gets
    the LLM facet with the client's parameters, and the bare construction is not a component on its own."""
    clients: dict[str, tuple[Signal, str]] = {}
    for info in symbols.values():
        for call in info.d.calls:
            if call.target:
                for s in info.signals:
                    if s.line == call.line and s.facet in ("client", "vector_db", "embedding", "reranker", "memory",
                                                           "retriever", "agent", "tool"):
                        clients.setdefault(call.target.split(".")[-1], (s, info.key))
    if not clients:
        return
    for info in symbols.values():
        for call in info.d.calls:
            parts = _parts(call)
            if len(parts) < 2:
                continue
            hit = clients.get(parts[-2]) or clients.get(parts[0])
            if hit is None or (hit[1] == info.key and hit[0].line == call.line):
                continue
            bound, via = hit
            facet = "llm" if bound.facet == "client" else bound.facet
            if facet == "llm" and parts[-1] not in LLM_METHODS | {"create", "parse"}:
                continue
            if facet == "llm" and any("embed" in p.lower() for p in parts[1:]):  # client.embeddings.create: not a chat call
                continue
            info.signals.append(Signal(facet, 0.9, call.line, call.snippet,
                                       f"uses {parts[-2] if len(parts) > 1 else parts[0]} ({bound.fact})",
                                       bound.provider, dict(bound.params), via=via))
    # a client nobody calls directly is used implicitly by a framework (e.g. LlamaIndex Settings.llm): it is the LLM
    used = set()
    for info in symbols.values():
        for call in info.d.calls:
            parts = _parts(call)
            if len(parts) >= 2:
                used.update({parts[-2], parts[0]})
    for info in symbols.values():
        for call in info.d.calls:
            if call.target and call.target.split(".")[-1] not in used:
                for s in info.signals:
                    if s.facet == "client" and s.line == call.line:
                        info.signals.append(Signal("llm", 0.9, s.line, s.snippet, f"{s.fact}, used implicitly by the framework",
                                                   s.provider, dict(s.params)))


def _registry_tools(files: list[FileFacts], symbols: dict[str, SymbolInfo]) -> None:
    """Functions dispatched through a name->function registry that some function indexes are tools."""
    registries = {r.name: (f.path, r) for f in files for r in f.registries}
    dispatched = {name for info in symbols.values() for name in info.d.subscripted if name in registries}
    by_name: dict[str, list[SymbolInfo]] = {}
    for info in symbols.values():
        by_name.setdefault(info.d.name, []).append(info)
    for name in dispatched:
        path, reg = registries[name]
        for fn in reg.entries.values():
            for info in by_name.get(fn, []):
                info.signals.append(Signal("tool", 0.9, reg.line, "", f"dispatched as a tool via {name}[...]"))
