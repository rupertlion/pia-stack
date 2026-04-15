import sys
import re
from langgraph.graph import StateGraph, END
from typing import TypedDict
import httpx
from qdrant_client import QdrantClient

LLM_URL = "http://localhost:8000/v1/chat/completions"
SEARCH_URL = "http://localhost:8080/search"
EMBED_URL = "http://localhost:8081/embed"
CRAWL_URL = "http://localhost:11235/crawl"
QDRANT_URL = "http://localhost:6333"

PROFILES = {
    "CODER": {
        "model": "qwen2.5-coder:14b",
        "system": "You are an expert software engineer. Write clean, well-documented code. Include error handling. Follow best practices. Be direct.",
        "tools": ["rag"], # <--- ADD "rag" HERE
    },
    "SHOPPER": {
        "model": "qwen3:32b",
        "system": "You are a deal-hunting research assistant. Compare prices across sources. Flag potential scams. Calculate total cost of ownership. Present findings in comparison tables. Be skeptical of too-good-to-be-true deals.",
        "tools": ["web", "crawl"],
    },
    "ANALYST": {
        "model": "qwen3:32b",
        "system": "You are a market intelligence analyst for an executive search firm. Track competitor movements, leadership changes, and market trends. Cross-reference local archives with live web data. Present findings as actionable intelligence briefings.",
        "tools": ["web", "crawl", "rag"],
    },
    "PERSONAL": {
        "model": "qwen3:32b",
        "system": "You are Rupert Lion's personal chief of staff. You have access to his email archives, calendar, and documents. You know his preferences, schedule, and priorities. Be proactive about surfacing relevant information. Respond as a trusted PA who knows the principal's world intimately.",
        "tools": ["rag"],
    },
    "THINKER": {
        "model": "qwen3:32b",
        "system": "You are a Socratic thought partner. Challenge assumptions. Explore trade-offs from multiple angles. Don't hedge or give generic advice. Push for specific, actionable conclusions. Think from first principles.",
        "tools": ["web"],
    },
}

ROUTER_PROMPT = """You are a query router. Classify the user's query into exactly ONE profile.

CODER: Writing, fixing, reviewing, or planning code. Mentions programming languages, scripts, bugs, APIs, databases, deployment.

SHOPPER: Comparing prices, finding deals, product research, marketplace queries. Mentions buying, cost, pricing, products, reviews, specifications.

ANALYST: Market intelligence, competitor tracking, industry research, executive moves, sector analysis, candidate sourcing.

PERSONAL: Personal schedule, emails, calendar, contacts, travel plans, health appointments, family logistics. References "my", "our", specific people by name, or asks about past communications.

THINKER: Open-ended strategy, technical architecture decisions, creative exploration, "how should I approach X", thought experiments, setup planning.

Respond with ONLY the profile name. Nothing else. No explanation."""


class AgentState(TypedDict):
    query: str
    profile: str
    local_context: list[str]
    web_context: list[str]
    response: str


def classify_profile(state):
    resp = httpx.post(LLM_URL, json={
        "model": "qwen3:32b",
        "messages": [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": state["query"]},
        ],
        "max_tokens": 500,
        "temperature": 0.1,
    }, timeout=300.0) # <--- INCREASED TIMEOUT HERE
    msg = resp.json()["choices"][0]["message"]

    raw_content = msg.get("content", "").strip()
    raw_reasoning = msg.get("reasoning", "").strip()

    # Method 1: Check content for a clean answer
    if raw_content:
        first_word = raw_content.split()[0].strip().upper().rstrip(".,!:")
        for name in PROFILES:
            if name == first_word:
                state["profile"] = name
                print("  Profile: " + name + " (from content)")
                return state

    # Method 2: Scan content and reasoning for profile keywords
    for source in [raw_content.upper(), raw_reasoning.upper()]:
        for name in ["CODER", "SHOPPER", "ANALYST", "PERSONAL"]:
            if name in source:
                state["profile"] = name
                print("  Profile: " + name + " (from scan)")
                return state

    # Method 3: Keyword fallback from the query itself
    query_lower = state["query"].lower()
    code_words = ["python", "function", "code", "script", "bug", "debug", "api",
                  "database", "deploy", "refactor", "class", "method", "import",
                  "javascript", "html", "css", "react", "sql", "git", "docker",
                  "write a", "fix the", "create a script", "build a"]
    shop_words = ["price", "pricing", "compare", "cost", "buy", "deal", "cheap",
                  "marketplace", "product", "review", "specs", "vs", "versus",
                  "shopping", "purchase", "discount"]
    analyst_words = ["market", "competitor", "industry", "executive", "hire",
                     "leadership", "sector", "trend", "intelligence", "firm",
                     "announced", "spencer stuart", "dhr", "heidrick", "korn ferry",
                     "candidate", "sourcing"]
    personal_words = ["my meeting", "my calendar", "my email", "kate", "lauren",
                      "cigna", "zepbound", "origins lodge", "costa rica",
                      "when is my", "what did i", "my schedule", "remind me"]

    for word in personal_words:
        if word in query_lower:
            state["profile"] = "PERSONAL"
            print("  Profile: PERSONAL (from keywords)")
            return state
    for word in code_words:
        if word in query_lower:
            state["profile"] = "CODER"
            print("  Profile: CODER (from keywords)")
            return state
    for word in shop_words:
        if word in query_lower:
            state["profile"] = "SHOPPER"
            print("  Profile: SHOPPER (from keywords)")
            return state
    for word in analyst_words:
        if word in query_lower:
            state["profile"] = "ANALYST"
            print("  Profile: ANALYST (from keywords)")
            return state

    # Default
    state["profile"] = "THINKER"
    print("  Profile: THINKER (default)")
    return state


def search_rag(state):
    profile = PROFILES.get(state["profile"], {})
    if "rag" not in profile.get("tools", []):
        return state
    try:
        # 1. Get the embedding for the query
        embed_resp = httpx.post(EMBED_URL, json={
            "inputs": [state["query"]], "truncate": True
        }, timeout=60.0)
        query_vector = embed_resp.json()[0]
        
        results = []
        # 2. Search each collection via direct REST API
        for collection in ["emails", "documents", "calendar", "codebase"]:
            try:
                search_resp = httpx.post(f"{QDRANT_URL}/collections/{collection}/points/search", json={
                    "vector": query_vector,
                    "limit": 5,
                    "score_threshold": 0.10,
                    "with_payload": True
                }, timeout=10.0)
                
                if search_resp.status_code == 200:
                    hits = search_resp.json().get("result", [])
                    for hit in hits:
                        print(f"  [DEBUG] Found in {collection} with score: {hit['score']:.3f}")
                        results.append(hit.get("payload", {}).get("content", ""))
                else:
                    print(f"  [DEBUG ERROR] {collection} failed: HTTP {search_resp.status_code}")
            except Exception as e:
                print(f"  [DEBUG ERROR] {collection} connection failed: {e}")
                
        state["local_context"] = results
        print("  RAG results: " + str(len(state["local_context"])))
    except Exception as e:
        print("  RAG error: " + str(e))
    return state


def search_web(state):
    profile = PROFILES.get(state["profile"], {})
    tools = profile.get("tools", [])
    if "web" not in tools:
        return state

    try:
        resp = httpx.get(SEARCH_URL, params={
            "q": state["query"], "format": "json",
            "categories": "general,news", "language": "en",
        }, timeout=30.0)
        results = resp.json().get("results", [])[:5]

        web_items = []
        for r in results:
            web_items.append("[" + r["title"] + "]\n" + r["content"] + "\nURL: " + r["url"])

        if "crawl" in tools:
            for r in results[:2]:
                try:
                    scrape_resp = httpx.post(CRAWL_URL, json={
                        "urls": [r["url"]], "priority": 10,
                    }, headers={"Authorization": "Bearer pia-local"}, timeout=45.0)
                    scrape_data = scrape_resp.json()
                    markdown = scrape_data.get("results", [{}])[0].get("markdown", "")
                    if markdown:
                        page_text = str(markdown) if markdown else ""
                        web_items.append("[FULL PAGE: " + r["title"] + "]\n" + page_text[:4000])
                except Exception:
                    pass

        state["web_context"] = web_items
        print("  Web results: " + str(len(web_items)))
    except Exception as e:
        print("  Web error: " + str(e))
    return state


def generate(state):
    profile = PROFILES.get(state["profile"], PROFILES["THINKER"])

    context_parts = []
    if state.get("local_context"):
        context_parts.append("LOCAL DATA:\n" + "\n---\n".join(state["local_context"]))
    if state.get("web_context"):
        context_parts.append("LIVE WEB DATA:\n" + "\n---\n".join(state["web_context"]))
    context = "\n\n".join(context_parts) if context_parts else ""

    user_msg = state["query"]
    if context:
        user_msg = "Context:\n" + context + "\n\nQuery: " + state["query"]

    resp = httpx.post(LLM_URL, json={
        "model": profile["model"],
        "messages": [
            {"role": "system", "content": profile["system"]},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }, timeout=300.0)
    msg = resp.json()["choices"][0]["message"]
    raw = msg.get("content", "")
    if not raw.strip():
        raw = msg.get("reasoning", "")
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    state["response"] = cleaned
    return state


graph = StateGraph(AgentState)
graph.add_node("classify", classify_profile)
graph.add_node("rag", search_rag)
graph.add_node("web", search_web)
graph.add_node("generate", generate)

graph.set_entry_point("classify")
graph.add_edge("classify", "rag")
graph.add_edge("rag", "web")
graph.add_edge("web", "generate")
graph.add_edge("generate", END)

app = graph.compile()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is exec search?"
    print("\nQuery: " + query)
    result = app.invoke({
        "query": query, "profile": "",
        "local_context": [], "web_context": [], "response": "",
    })
    print("\n" + "=" * 60)
    print("[" + result["profile"] + " profile | " + PROFILES[result["profile"]]["model"] + "]")
    print("=" * 60)
    print(result["response"])
