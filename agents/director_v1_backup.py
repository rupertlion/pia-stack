import sys
from langgraph.graph import StateGraph, END
from typing import TypedDict
import httpx
from router_prompt import ROUTER_SYSTEM


class AgentState(TypedDict):
    query: str
    route: str
    local_context: list[str]
    web_context: list[str]
    response: str


def classify_query(state: AgentState) -> AgentState:
    resp = httpx.post("http://localhost:8000/v1/chat/completions", json={
        "model": "qwen3:32b",
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": state["query"]}
        ],
        "max_tokens": 500,
        "temperature": 0.1,
    }, timeout=120.0)
    raw = resp.json()["choices"][0]["message"]["content"]

    import re
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    route = "DIRECT"
    for label in ["HYBRID", "LOCAL", "WEB", "DIRECT"]:
        if label in cleaned.upper():
            route = label
            break

    state["route"] = route
    print(f"  Route: {route}")
    return state


def search_local(state: AgentState) -> AgentState:
    embed_resp = httpx.post("http://localhost:8081/embed", json={
        "inputs": [state["query"]], "truncate": True
    }, timeout=60.0)
    query_vector = embed_resp.json()[0]

    from qdrant_client import QdrantClient
    qclient = QdrantClient(url="http://localhost:6333")

    results = []
    for collection in ["emails", "documents", "calendar"]:
        try:
            hits = qclient.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=5,
                score_threshold=0.65,
            )
            results.extend(hits)
        except Exception:
            pass

    state["local_context"] = [
        r.payload.get("content", "") for r in results
    ]
    print(f"  Local results: {len(state['local_context'])}")
    return state


def search_web(state: AgentState) -> AgentState:
    # Step 1: Get search results from SearXNG
    resp = httpx.get("http://localhost:8080/search", params={
        "q": state["query"],
        "format": "json",
        "categories": "general,news",
        "language": "en",
    }, timeout=30.0)
    results = resp.json().get("results", [])[:5]

    web_items = []
    for r in results:
        web_items.append(f"[{r['title']}]\n{r['content']}\nURL: {r['url']}")

    # Step 2: Scrape the top 2 results for full content via Crawl4AI
    for r in results[:2]:
        try:
            scrape_resp = httpx.post("http://localhost:11235/crawl", json={
                "urls": [r["url"]],
                "priority": 10,
            }, headers={"Authorization": "Bearer pia-local"}, timeout=45.0)
            scrape_data = scrape_resp.json()
            markdown = scrape_data.get("results", [{}])[0].get("markdown", "")
            if markdown:
                # Truncate to avoid blowing context window
                page_text = str(markdown) if markdown else ""
                web_items.append(f"[FULL PAGE: {r['title']}]\n{page_text[:4000]}")
        except Exception as e:
            print(f"  Scrape failed for {r['url']}: {e}")

    state["web_context"] = web_items
    print(f"  Web results: {len(web_items)} (incl. scraped pages)")
    return state


def generate_response(state: AgentState) -> AgentState:
    context_parts = []
    if state.get("local_context"):
        context_parts.append(
            "LOCAL DATA:\n" + "\n---\n".join(state["local_context"])
        )
    if state.get("web_context"):
        context_parts.append(
            "LIVE WEB DATA:\n" + "\n---\n".join(state["web_context"])
        )
    context = "\n\n".join(context_parts) if context_parts else "No additional context."

    resp = httpx.post("http://localhost:8000/v1/chat/completions", json={
        "model": "qwen3:32b",
        "messages": [
            {"role": "system", "content": (
                "You are a Private Intelligence Analyst. "
                "Use the provided context to answer precisely. "
                "Cite whether info came from local archives or live web. "
                "Be direct, concise, and actionable."
            )},
            {"role": "user", "content": (
                f"Context:\n{context}\n\nQuery: {state['query']}\n\n/no_think"
            )}
        ],
        "max_tokens": 2048,
        "temperature": 0.3
    }, timeout=300.0)
    raw = resp.json()["choices"][0]["message"]["content"]

    # Strip thinking tags if present
    import re
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    state["response"] = cleaned
    return state


def route_decision(state: AgentState) -> str:
    route = state.get("route", "DIRECT")
    if route == "LOCAL":
        return "local"
    elif route == "WEB":
        return "web"
    elif route == "HYBRID":
        return "hybrid"
    return "direct"


graph = StateGraph(AgentState)
graph.add_node("classify", classify_query)
graph.add_node("local_search", search_local)
graph.add_node("web_search", search_web)
graph.add_node("generate", generate_response)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route_decision, {
    "local": "local_search",
    "web": "web_search",
    "hybrid": "local_search",
    "direct": "generate",
})
graph.add_edge("web_search", "generate")
graph.add_conditional_edges("local_search",
    lambda s: "web" if s["route"] == "HYBRID" else "generate",
    {"web": "web_search", "generate": "generate"}
)
graph.add_edge("generate", END)
app = graph.compile()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is exec search?"
    print(f"\nQuery: {query}")
    result = app.invoke({
        "query": query, "route": "",
        "local_context": [], "web_context": [], "response": "",
    })
    print(f"\n{'='*60}")
    print(result["response"])
