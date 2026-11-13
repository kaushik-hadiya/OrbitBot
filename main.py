"""Stage 5: Self-RAG retrieval decision + document grading + support check + usefulness check
with query rewriting, over the Orbitra knowledge base.
Retrieval is hybrid: vector + BM25 keyword search, merged with RRF, then reranked by a cross-encoder.

Each turn starts with start_turn (a follow-up is rewritten into a standalone question using the conversation)
and ends with finish (the answer is added to the conversation, kept per thread_id by chat_app's checkpointer).

Flow: question -> decide_retrieval -> (knowledge_base) plan_queries -> retrieve -> grade_documents
                                   -> (action)         agent <-> tools (search_kb + agent_tools.py) -> answer
                                   -> (small_talk)     generate_direct -> answer
                                   -> (distress)       support_wellbeing -> fixed reply, no LLM call
                                   -> (out_of_scope)   decline_out_of_scope -> fixed reply, no LLM call
      grade_documents -> (some relevant) generate -> check_support
                      -> (none relevant, rewrite left) rewrite_query -> retrieve
                      -> (none relevant, no rewrite left) suggest_contact -> answer
      check_support -> (fully supported) check_useful
                    -> (not supported, retry left) revise_answer -> check_support
                    -> (not supported, no retry left) flag_unsupported -> answer
      check_useful -> (useful) answer
                   -> (not useful, rewrite left) rewrite_query -> retrieve
                   -> (not useful, no rewrite left) suggest_contact -> answer
"""
import argparse
import hashlib
import json
import re
import os
import uuid
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import yaml
from dotenv import load_dotenv
from fastembed.rerank.cross_encoder import TextCrossEncoder
#from groq import BadRequestError
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
#from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel, Field

import agent_tools

from openai import BadRequestError
from langchain_openai import ChatOpenAI
import os

load_dotenv()

DOCS_DIR = Path("documents")
MODELS_DIR = "models"
CHROMA_DIR = "chroma_db"
COLLECTION = "orbitra_kb"

# load the documents and parse the YAML header functions
def parse_markdown(path: Path) -> Document:
    """Read one .md file and turn its YAML front-matter into metadata."""
    text = path.read_text(encoding="utf-8")
    metadata = {"source": str(path)}

    if text.startswith("---"):
        _, front_matter, text = text.split("---", 2)
        try:
            header = yaml.safe_load(front_matter)
        except yaml.YAMLError:
            # e.g. HR-POSH-001 has an unquoted ": " inside a value, so read plain "key: value" lines
            lines = [line.split(":", 1) for line in front_matter.strip().splitlines() if ":" in line]
            header = {k.strip(): v.strip() for k, v in lines}
        for key, value in header.items():
            if value is None:
                continue
            # Chroma only stores str/int/float/bool, so flatten lists and dates
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            elif not isinstance(value, (str, int, float, bool)):
                value = str(value)
            metadata[key] = value

    return Document(page_content=text.strip(), metadata=metadata)


def is_loaded(doc: Document) -> bool:
    """Keep out of the index what OrbitBot must never answer from:
    - superseded versions: a newer one exists, so they would only take search slots from it
    - Confidential documents restricted to particular teams (SAL-001 is for Sales, CS and Finance only);
      a Confidential document meant for all employees (HR-COMP-001) stays in"""
    meta = doc.metadata
    if meta.get("status") == "superseded":
        return False
    restricted = "all employees" not in str(meta.get("applies_to", "")).lower()
    return not (meta.get("classification") == "Confidential" and restricted)


def load_documents() -> list[Document]:
    docs = [parse_markdown(p) for p in sorted(DOCS_DIR.rglob("*.md"))]
    return [d for d in docs if is_loaded(d)]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# ONNX model, downloaded into models/ on the first run (bge vectors come out normalized).
# batch_size: default is 256, which needs >1.5 GB RAM for these chunks and froze the laptop.
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", cache_dir=MODELS_DIR, batch_size=16)


def chunk_id(chunk: Document) -> str:
    """A fingerprint of the chunk's text and metadata: any edit to a document changes its chunks' ids."""
    payload = json.dumps([chunk.page_content, chunk.metadata], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# embeddings and the vectorstore (build once, and then load)
def get_vectorstore(chunks: list[Document]) -> Chroma:
    """Load the saved Chroma DB, or (re)build it if it is missing, incomplete or out of date."""
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # ids are content fingerprints, so a different set means first run, a build that crashed halfway,
    # or documents that were edited, added or removed (a count check alone misses edits)
    ids = [chunk_id(c) for c in chunks]
    stored = set(vectorstore.get(include=[])["ids"])
    if stored == set(ids):
        return vectorstore
    if stored:
        print(f"Vectorstore is out of date ({len(stored)} stored chunks, {len(ids)} expected), rebuilding...")
        vectorstore.reset_collection()

    print(f"Building vectorstore from {len(chunks)} chunks...")
    step = embeddings.batch_size
    for i in range(0, len(chunks), step):
        vectorstore.add_documents(chunks[i:i + step], ids=ids[i:i + step])
        print(f"\r  embedded {min(i + step, len(chunks))}/{len(chunks)} chunks", end="", flush=True)
    print()
    return vectorstore


chunks = splitter.split_documents(load_documents())
vectorstore = get_vectorstore(chunks)


# hybrid search: vector + BM25 keyword candidates, merged with RRF, then reranked
# keyword search catches exact words like "comp-off" that the embedding model misses
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


CANDIDATES = 12
TOP_K = 4
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": CANDIDATES})
bm25_retriever = BM25Retriever.from_documents(chunks, k=CANDIDATES, preprocess_func=tokenize)

# a cross-encoder reads the query and each chunk together, so it judges relevance much better
# than comparing two separate embeddings; ~130 MB ONNX model, downloaded into models/
reranker = TextCrossEncoder("jinaai/jina-reranker-v1-turbo-en", cache_dir=MODELS_DIR)


def reciprocal_rank_fusion(*ranked_lists: list[Document], k: int = 60) -> list[Document]:
    """Merge rankings: a chunk scores 1/(k + rank) in every list it appears in."""
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, 1):
            key = (doc.metadata["source"], doc.page_content)
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


# every chunk is Orbitra's, but few say so, and the reranker scores the word itself: for "What is the notice
# period at Orbitra?" it ranked chunks that mention Orbitra above HR-EXIT-001's notice table, which doesn't.
# Naming the company on every chunk makes that match equal for all of them. (Removing the name from the query
# instead broke questions about the company itself, e.g. "Tell me about Orbitra".)
RERANK_PREFIX = "Orbitra Technologies\n"


# fastembed's reranker isn't a LangChain component, so it sends nothing to LangSmith on its own:
# @traceable makes it (and hybrid_search, one per sub-question) show up as its own timed step
@traceable(name="rerank")
def rerank(query: str, candidates: list[Document]) -> list[float]:
    return list(reranker.rerank(query, [RERANK_PREFIX + d.page_content for d in candidates], batch_size=4))


@traceable(run_type="retriever", name="hybrid_search")
def hybrid_search(query: str, top_k: int = TOP_K) -> list[Document]:
    candidates = reciprocal_rank_fusion(vector_retriever.invoke(query), bm25_retriever.invoke(query))[:CANDIDATES]
    scores = rerank(query, candidates)
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# llm and prompt
# max_retries: on a 429, the client waits as provider says, then retries
# OpenRouter speaks the OpenAI API, so ChatOpenAI just points at its URL
OPENROUTER = dict(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=6,
)
llm = ChatOpenAI(**OPENROUTER)
# yes/no judgments don't need deep thinking: ask OpenRouter for low reasoning effort
judge_llm = ChatOpenAI(**OPENROUTER, extra_body={"reasoning": {"effort": "low"}})
# max_retries only covers HTTP errors (429/5xx), so two bad replies need their own retry (up to 3 tries):
# - BadRequestError: the judge returned a value outside the schema (e.g. "partial" instead of
#   "partially") and the provider rejected it with a 400
# - ValueError: the provider failed mid-answer and OpenRouter sent back an empty 200
#   (finish_reason 'error'), so LangChain found no parsed output
RETRY_BAD_OUTPUT = dict(retry_if_exception_type=(BadRequestError, ValueError), stop_after_attempt=3)


def require_text(message: AIMessage) -> str:
    """The reply's text; an empty reply (the same empty 200 as above, on a plain-text call) raises ValueError,
    so .with_retry(**RETRY_BAD_OUTPUT) asks again instead of passing on a blank answer or search query."""
    text = message.content.strip()
    if not text:
        raise ValueError("the LLM returned an empty reply")
    return text


def text_chain(chat_prompt: ChatPromptTemplate, model: ChatOpenAI):
    """prompt -> model -> non-empty text, retried on bad output like the structured-output chains."""
    return (chat_prompt | model | RunnableLambda(require_text)).with_retry(**RETRY_BAD_OUTPUT)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are OrbitBot, an internal assistant for Orbitra Technologies employees. "
     "Answer the question using only the context below. Give the direct answer first. "
     "Then add, one short line each, only the conditions, exceptions and next steps that "
     "change or limit that answer for what was asked (e.g. who must approve it, a deadline "
     "for it, an exception to it, what to do if it doesn't work). Leave out related rules "
     "that don't change the answer (other leave types, carry-over, payouts, general "
     "process steps, contacts), and never add a line saying there are no other conditions. "
     "If nothing changes the answer, the direct answer alone is enough. When the answer is no, "
     "don't suggest a way around it: mention an exception, approval or alternative only if the "
     "context says it applies to the user's case (e.g. not a fix-only exception for a feature, or a "
     "tool approved for less sensitive data than theirs). If the context does not "
     "contain the answer, say you don't know; if it answers only some parts of the question, "
     "answer those and say which part you don't know. If chunks from different policies "
     "touch the same topic, follow the policy whose subject is the question's topic and "
     "put its process first; never carry a rule from a policy about something else (another "
     "team, ticket type, region or group) into the answer. If the answer depends on details the "
     "user did not give (location, level, etc.), list the options from the context "
     "and ask which applies. Cite the doc_id of every source you use, like "
     "[HR-LV-001].\n\nContext:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs: list[Document]) -> str:
    # the title tells the LLMs what a chunk's rules are for: a chunk from the middle of a document
    # has no heading of its own (e.g. IT-001's ticket SLAs vs CS-001's customer SLAs)
    return "\n\n".join(f"[{d.metadata.get('doc_id', '?')}: {d.metadata.get('title', '?')}] {d.page_content}"
                       for d in docs)


# routing decision (Self-RAG's "Retrieve" step plus a scope check), returned as a Pydantic object
class RouteDecision(BaseModel):
    """Decide how OrbitBot should handle the question."""
    route: Literal["knowledge_base", "action", "small_talk", "distress", "out_of_scope"] = Field(description=
        "'knowledge_base' for anything an employee could ask about Orbitra or their work there (policies, "
        "benefits, processes, tools, data handling, people, products, company facts), even if the documents "
        "may not cover it; 'action' when the user asks OrbitBot to do something for them or to look up live or "
        "personal data (their own leave balance, a colleague's contact details, similar past tickets, raising a "
        "ticket, resetting, booking, applying for or approving something, or a calculation); 'small_talk' for "
        "greetings, thanks or what OrbitBot can do; 'distress' when the user says they themselves are struggling "
        "emotionally; 'out_of_scope' only if the topic has nothing to do with Orbitra or working there.")
    reason: str = Field(description="One short sentence explaining the decision.")


decide_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You route questions sent to OrbitBot, the internal assistant of Orbitra Technologies. Its "
     "company knowledge base covers HR, IT, security, finance, admin, legal, engineering, product and "
     "sales documents. Choose knowledge_base for anything about Orbitra or how things work there, even "
     "if it sounds general (e.g. 'how do I reset my password?'). This includes employee benefits, pay, "
     "workplace tools and how work data may be handled. You cannot see the documents, so never reject a "
     "question because you think they may not cover it; judge only its topic. Choose small_talk only for "
     "greetings, thanks, brief pleasantries, or questions about what OrbitBot can do. Choose out_of_scope "
     "only if the topic has nothing to do with Orbitra or working there, such as general knowledge, "
     "personal plans, or writing and coding help. Examples of out_of_scope: 'plan a trip to Jaipur', "
     "'give me the constitution of India'. Examples of knowledge_base: 'what's the hotel limit for a "
     "business trip to Jaipur?', 'does Orbitra pay for a gym membership?'. "
     "Choose action when the user wants OrbitBot to do something or look something up for them rather than "
     "explain a rule: 'how many ELs do I have left?', 'what's Priya Nair's email?', 'reset my password for me', "
     "'raise a ticket for my broken screen', 'book me a flight'. A how-to question ('how do I reset my "
     "password?') is knowledge_base, and so is describing a problem and asking what happens, what the rules are "
     "or what to do ('I lost my laptop, what happens now?'): choose action only when the user asks OrbitBot "
     "itself to act or to look up their own or a colleague's data. Choose distress only when the user describes their own emotional state, "
     "such as feeling hopeless, unable to cope, overwhelmed or unable to sleep from stress. Reports of "
     "harassment, fraud, threats or security incidents are knowledge_base, even when the user is upset, "
     "because the policies say who to contact. When unsure, choose knowledge_base."),
    ("human", "{question}"),
])
decider =decide_prompt | judge_llm.with_structured_output(RouteDecision).with_retry(**RETRY_BAD_OUTPUT)

direct_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are OrbitBot, an internal assistant for Orbitra Technologies employees. Answer briefly. "
     "This question was judged not to need company documents, so do not state any Orbitra-specific "
     "facts or policies; if the user needs those, suggest they ask about them directly."),
    ("human", "{question}"),
])

# fixed text, so refusing an out-of-scope question costs no LLM call beyond the decider
OUT_OF_SCOPE_REPLY = ("I can only help with questions about Orbitra and working here, such as HR, IT, "
                      "finance, security, legal and company policies. Please ask me about one of those.")

# fixed text, so a safety answer never depends on retrieval or generation (EAP details from HR-BEN-001)
WELLBEING_REPLY = (
    "I'm sorry you're going through this, and I'm glad you reached out. You don't have to handle it alone.\n\n"
    "- **Talk to a counsellor:** Orbitra's Employee Assistance Programme gives free, confidential counselling "
    "with MindHaven, 24x7, at **1800-102-7777** or in the MindHaven app (6 sessions a year for you and each "
    "family member, India) [HR-BEN-001]. Outside India, your HR Business Partner can connect you with support.\n"
    "- **If you might harm yourself or are in danger right now,** call your local emergency number (112 in "
    "India, 911 in the US, 999 in the UK) or a crisis line: Tele-MANAS **14416** (India), **988** (US), "
    "Samaritans **116 123** (UK).\n\n"
    "You can also talk to your manager or HRBP. When you're ready, I can help with leave or other support "
    "Orbitra offers.")


# query planning: split a question into the separate facts it needs (one search each)
class QueryPlan(BaseModel):
    """The searches needed to answer the question."""
    sub_questions: list[str] = Field(description="1 to 3 standalone search questions, one per thing the "
        "answer needs to look up; just the original question if it asks one thing that one policy answers.")


plan_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You plan searches over Orbitra's internal knowledge base (HR, IT, security, finance, admin, legal, "
     "engineering, product and sales policies). Write one search question for each separate thing the "
     "answer has to look up (at most 3).\n"
     "Split when:\n"
     "- the question asks two or more things (\"X and Y\", \"what will happen and what do I do\"), or\n"
     "- answering it means combining rules that usually sit in different policies, for example:\n"
     "  - whether something can happen on a date -> the rule for it (e.g. deployment windows and change "
     "freezes) and the holiday calendar\n"
     "  - an expense claim or purchase -> the spending limit and who approves it (the approval matrix)\n"
     "  - when a bonus or reimbursement will be paid -> the rule for it and the payroll cut-off dates\n"
     "  - resigning or leaving -> the notice period, leave encashment, and what happens to each benefit "
     "asked about (e.g. ESOP vesting, gratuity)\n"
     "  - a new joiner or someone on probation -> the onboarding/probation rules and the policy asked about "
     "(e.g. leave)\n"
     "  - a lost or stolen device -> the IT asset-loss rules and how to report a security incident\n"
     "Do not split a question that asks one thing answered by one policy: return it unchanged as the only "
     "item.\n"
     "Each sub-question is a plain, standalone search question about one topic. Keep the user's own key "
     "terms and specifics as they wrote them (e.g. \"comp-off\", \"40 days after birth\"): search matches "
     "exact words. Never name or guess which policy holds the answer (no \"according to the X policy\"). "
     "Keep the user's region and level only if they gave them; never add placeholders like "
     "\"region: unspecified\" or notes in brackets. Leave out the user's own amounts and calculations."),
    ("human", "{question}"),
])
planner = plan_prompt | judge_llm.with_structured_output(QueryPlan).with_retry(**RETRY_BAD_OUTPUT)

MAX_SUB_QUESTIONS = 3
TOP_K_PER_PART = 3   # chunks per sub-question when there are several (one question keeps TOP_K = 4)


# document grading (Self-RAG's "IsRel" step), one grade per retrieved chunk
class DocGrade(BaseModel):
    """Relevance grade for one retrieved chunk."""
    is_relevant: bool = Field(description="True if the chunk contains information that helps answer "
        "the question; False if it is off-topic, superseded, or for a different region or group "
        "than the question asks about.")
    reason: str = Field(description="One short sentence explaining the grade.")


grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether a chunk from Orbitra's knowledge base is relevant to an employee's question. "
     "Use the chunk's metadata: a chunk with status 'superseded' is never relevant, because a newer "
     "version exists. A chunk whose applies_to clearly excludes the region or group the question is "
     "about is not relevant. If the question names no region, any region can be relevant. Otherwise "
     "be lenient: the chunk only needs to contain part of the answer. If the question has several "
     "parts, a chunk that answers any one part is relevant."),
    ("human",
     "Question: {question}\nParts of the question: {parts}\n\nChunk metadata: doc_id={doc_id}, title={title}, status={status}, "
     "applies_to={applies_to}\n\nChunk:\n{chunk}"),
])
grader = grade_prompt | judge_llm.with_structured_output(DocGrade).with_retry(**RETRY_BAD_OUTPUT)


# support check (Self-RAG's "IsSup" step): is the answer grounded in the chunks?
class SupportGrade(BaseModel):
    """How well the retrieved chunks support the generated answer."""
    # filled first, so the model checks each suggested way out against the user's case before grading
    # (without it, workarounds like "get VP approval" for a fix-only freeze passed as 'fully')
    alternatives_check: list[str] = Field(description="Before grading: each exception, approval, "
        "workaround or alternative the answer suggests, followed by the user's case and whether a chunk "
        "says it applies to that case (quote the chunk's limit if it doesn't). Empty if none.")
    support: Literal["fully", "partially", "none"] = Field(description="'fully' if every factual "
        "claim in the answer is stated in, or directly calculated from, the chunks; 'partially' if "
        "some claims are not; 'none' if the answer is mostly unsupported.")
    unsupported_claims: list[str] = Field(description="Each claim not backed by the chunks, including "
        "every alternative from alternatives_check that doesn't apply to the user's case; empty if "
        "fully supported.")


support_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether an answer from OrbitBot is grounded in the retrieved chunks from Orbitra's "
     "knowledge base. A claim is supported if a chunk states it, if the user stated it in their "
     "question (e.g. their own salary, dates or counts), or if it is a direct calculation or "
     "rewording of those. Saying 'I don't know', asking the user a clarifying question, and "
     "citations like [HR-LV-001] are not claims. Anything else, such as numbers, names, rules or "
     "steps not found in the chunks or the question, is unsupported. A rule or step that is in a "
     "chunk is still unsupported if the answer applies it to something that chunk is not about "
     "(e.g. one team's, ticket type's, region's or group's process presented as applying to another); "
     "each chunk is prefixed with its doc_id, so check which document a claim comes from. Likewise, "
     "a suggested exception, approval, workaround or alternative is unsupported unless a chunk says "
     "it applies to the user's case; if a chunk limits it to other cases (other data classes, "
     "incident types, situations), it is unsupported even though the chunk mentions it.\n\n"
     "Chunks:\n{context}"),
    ("human", "Question: {question}\n\nAnswer to check:\n{answer}"),
])
# medium effort: one call per answer (not per chunk like the grader), and at low effort the check
# missed most workarounds that a chunk rules out for the user's case
support_llm = ChatOpenAI(**OPENROUTER, extra_body={"reasoning": {"effort": "medium"}})
support_checker = support_prompt | support_llm.with_structured_output(SupportGrade).with_retry(**RETRY_BAD_OUTPUT)

# used when the support check fails: edit the previous answer instead of starting over
revise_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are OrbitBot, an internal assistant for Orbitra Technologies employees. Your previous "
     "answer contained claims that the context does not support. Remove or correct only those "
     "claims and keep the rest of the answer as it was. If an unsupported claim suggests an exception, "
     "approval, workaround or alternative, delete it; don't replace it with a different one. Do not add "
     "any new details, even ones that seem helpful. Keep the doc_id citations like [HR-LV-001]."
     "\n\nContext:\n{context}"),
    ("human", "Question: {question}\n\nPrevious answer:\n{answer}\n\nUnsupported claims:\n{claims}"),
])

MAX_GENERATIONS = 2   # first answer + one revision


# usefulness check (Self-RAG's "IsUse" step): does the answer address the question?
class UsefulnessGrade(BaseModel):
    """Whether the answer actually addresses the user's question."""
    is_useful: bool = Field(description="True if the answer gives the user what they asked for, or asks "
        "a clarifying question with the options from the documents; False if it says the information "
        "is not available or answers a different question.")
    reason: str = Field(description="One short sentence explaining the grade.")


useful_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether OrbitBot's answer is useful to the employee who asked the question. Judge only "
     "whether it addresses what was asked, not whether it is correct; correctness was already checked."),
    ("human", "Question: {question}\n\nAnswer:\n{answer}"),
])
usefulness_checker = useful_prompt | judge_llm.with_structured_output(UsefulnessGrade).with_retry(**RETRY_BAD_OUTPUT)

# query rewriting: a new search query for retrieve only; grading and generation keep the original question
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite an employee's question into a search query for Orbitra's internal knowledge base "
     "(HR, IT, finance, admin, legal, engineering policies). The search is semantic, over policy "
     "documents. Never copy the user's own amounts, salaries, counts or dates into the query, and "
     "drop any calculation; keep the region/location; use the words a "
     "policy document would use (e.g. 'entitlement', 'leave types', 'notice period'). "
     "Return only the query."),
    ("human", "Question: {question}\n\nThis search query did not find the answer: {previous_query}"),
])
rewriter = text_chain(rewrite_prompt, judge_llm)

MAX_REWRITES = 1


# when the knowledge base can't answer: point to the owner of the closest document (from its metadata)
class ContactChoice(BaseModel):
    """Pick the document whose owning team can best help with the question."""
    doc_id: str | None = Field(description="doc_id of the retrieved document closest in topic to the "
        "question, so its owner can help; None if none of them is about the same topic.")


contact_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "OrbitBot could not answer an employee's question from Orbitra's knowledge base. From the "
     "documents that were retrieved, pick the one whose owning team is the best contact for this "
     "question (same topic, e.g. benefits, leave, IT access). Pick None if none is about the same topic."),
    ("human", "Question: {question}\n\nRetrieved documents:\n{documents}"),
])
contact_picker = contact_prompt | judge_llm.with_structured_output(ContactChoice).with_retry(**RETRY_BAD_OUTPUT)


# the action route: a tool-calling agent (ReAct loop: agent -> tools -> agent -> ... -> answer)
@tool
def search_kb(query: str) -> str:
    """Search Orbitra's knowledge base (HR, IT, security, finance, admin, legal, engineering, product and sales
    policies and how-to guides). Returns the best passages, each starting with [doc_id: title].
    Use one search per topic, in policy wording (e.g. 'MFA reset process', 'earned leave carry forward')."""
    return format_docs(hybrid_search(query)) or "Nothing found."


AGENT_TOOLS = [search_kb, *agent_tools.TOOLS]
MAX_TOOL_ROUNDS = 4   # then the agent must answer with what it has
agent_llm = llm.bind_tools(AGENT_TOOLS)
final_llm = llm.bind_tools(AGENT_TOOLS, tool_choice="none")   # same tools in the history, but no more calls

AGENT_SYSTEM = (
    "You are OrbitBot, an internal assistant for Orbitra Technologies employees. {user}\n\n"
    "Answer with your tools. Every company rule, number, name, contact or step you state must come from a tool "
    "result: use search_kb for any policy, process or how-to and cite its doc_ids like [HR-LV-001] (cite only "
    "doc_ids, never tool names). Before drafting a ticket, search_kb for the issue, so you can tell the user "
    "what to do first and what the ticket needs. "
    "Your tools can only read data and draft tickets. You cannot reset passwords, apply for or approve leave, "
    "book travel, approve expenses or submit tickets. When asked to do one of those, say plainly that you can't "
    "do it yourself, then use search_kb to give the user the steps or the self-service route, and if a ticket "
    "is the right way, prepare one with draft_ticket and tell the user to submit it. Never say or imply that "
    "anything was done, raised, created, submitted, reset, booked, applied for or approved. "
    "Share only the work contact details the directory returns; never guess anyone's private details. "
    "Give the direct answer first and keep it short."
)


AGENT_ANSWER_NOW = ("No more tool calls are possible. Answer my question now from the tool results above; "
                    "if they don't contain the answer, say you don't know.")
AGENT_EMPTY_REPLY = ("I couldn't find an answer to this with the information I can access. Please raise an "
                     "OrbitDesk ticket with the relevant team.")


def describe_user(employee_id: str | None) -> str:
    emp = agent_tools.EMPLOYEES_BY_ID.get(employee_id or "")
    if not emp:
        return ("The user is not signed in, so you don't know who they are and their personal data (such as "
                "leave balances) is unavailable.")
    return (f"The user is {emp['name']} ({emp['employee_id']}), {emp['title']} in {emp['department']}, "
            f"based in {emp['location']}.")


# conversation memory: a follow-up ("what about the US?") is rewritten into a standalone question first,
# so routing, search, grading and the checks all keep working on one self-contained question
MAX_HISTORY_MESSAGES = 6   # the last 3 exchanges are enough context for a follow-up
HISTORY_CHARS = 800        # per message: long answers are cut, the start carries the topic

# two fields, because one free-text rewrite kept mixing them up: it dropped facts about the user (their
# balance, level) and copied policy figures from OrbitBot's answers into the question, where the support
# check would then accept them as "stated by the user" without checking them against the documents
class StandaloneQuestion(BaseModel):
    """The latest message, made understandable without the conversation."""
    question: str = Field(description="The latest message rewritten to stand on its own: fill in the topic, "
        "region, leave type, person or thing it refers to ('it', 'she', 'what about...', 'and for...'). Keep "
        "the user's own words and intent. Never include company rules, limits or policy figures from OrbitBot's "
        "earlier answers. Unchanged if it already stands on its own or starts a new topic.")
    user_facts: list[str] = Field(description="Facts about the user themselves, from earlier in the "
        "conversation, that the answer depends on: e.g. 'I am an L4 engineer in Pune', 'I have 18.5 days of "
        "Earned Leave left'. Only facts about the user (their location, level, role, balances, amounts), never "
        "company rules or earlier questions and answers. Empty if the message is not about their own situation, "
        "if none are known, and always for thanks, greetings and other messages that ask nothing.")


condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You prepare the latest message in a conversation with OrbitBot, Orbitra's internal assistant, so it can "
     "be answered without the conversation. Don't answer it. Examples:\n"
     "1. User: 'How many sick days do UK employees get?' OrbitBot: '20 working days.' Latest: 'and in India?'\n"
     "   -> question 'How many sick days do employees in India get?', user_facts []\n"
     "2. User: 'My EL balance?' OrbitBot: 'You have 18.5 days of EL left.' Latest: 'Can I take 20 days off in "
     "December?'\n"
     "   -> question 'Can I take 20 days of Earned Leave in December?', user_facts ['I have 18.5 days of Earned "
     "Leave left']\n"
     "3. User: 'I'm L4 in Pune. Hotel limit in Mumbai?' OrbitBot: 'INR 7,000 a night.' Latest: 'Can I book a "
     "9000 rupee hotel then?'\n"
     "   -> question 'Can I book a 9,000 rupee per night hotel in Mumbai?', user_facts ['I am L4, based in "
     "Pune'] (the 7,000 limit is a company rule from OrbitBot's answer: it goes nowhere)\n"
     "4. User: 'Who approves a 2 lakh purchase?' OrbitBot: 'Your L8 Director.' Latest: 'thanks!'\n"
     "   -> question 'thanks!', user_facts []"),
    ("human", "Conversation so far:\n{history}\n\nLatest message: {message}"),
])
# medium effort: at low effort it left follow-ups like "and in India?" unchanged and copied policy limits into
# the question; one call per follow-up turn only (the first message of a conversation skips it)
condenser = (condense_prompt | support_llm.with_structured_output(StandaloneQuestion)
             .with_retry(**RETRY_BAD_OUTPUT))


def condense(history: list[AnyMessage], message: str) -> tuple[str, str]:
    """(question with the user's facts, question alone): the nodes that judge or answer need the facts, but
    the search only needs the question (the facts' amounts and balances are noise to BM25 and the vectors)."""
    result = condenser.invoke({"history": format_history(history), "message": message})
    question = result.question.strip() or message
    facts = [f.strip().rstrip(".") for f in result.user_facts if f.strip()]
    return (f"{question} ({'; '.join(facts)})" if facts else question), question


def turn_start(messages: list[AnyMessage]) -> int:
    """Index of the current turn's user message (the last HumanMessage)."""
    return max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)


def recent_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The last few user/assistant messages of earlier turns (finish leaves nothing else in the state)."""
    return [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-MAX_HISTORY_MESSAGES:]


def format_history(messages: list[AnyMessage]) -> str:
    return "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'OrbitBot'}: {m.content[:HISTORY_CHARS]}"
                     for m in messages)


# the graph(state,nodes,edges)
class GraphState(TypedDict):
    question: str  # the user's message, then (from start_turn on) its standalone version
    search_question: str  # the standalone question without the user's facts: what a one-part plan searches
    employee_id: str | None  # the signed-in user, if any (only get_leave_balance reads it)
    route: str  # knowledge_base, action, small_talk, distress or out_of_scope
    route_reason: str
    # the conversation: one user and one OrbitBot message per earlier turn, plus the current turn's
    # messages (the agent's tool calls and results while it works; finish removes them)
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    documents: list[Document]
    doc_grades: list[dict]
    answer: str
    generation_attempts: int
    support: str
    unsupported_claims: list[str]
    sub_questions: list[str]  # from plan_queries: one search per part
    search_query: str  # set by a rewrite; retrieve then searches only this
    rewrite_count: int
    is_useful: bool
    usefulness_reason: str


# with a checkpointer, every field survives into the next turn, so the per-turn ones start fresh here
# (otherwise e.g. last turn's search_query would replace this turn's search, and rewrite_count would block a rewrite)
TURN_RESET = {"documents": [], "doc_grades": [], "answer": "", "generation_attempts": 0, "support": "",
              "unsupported_claims": [], "sub_questions": [], "search_query": "", "rewrite_count": 0,
              "is_useful": False, "usefulness_reason": "", "tool_rounds": 0}


def start_turn(state: GraphState) -> dict:
    message = state["question"]
    history = recent_history(state.get("messages", []))
    question = search_question = message
    if history:   # the first message of a conversation has nothing to refer back to: no LLM call
        question, search_question = condense(history, message)
    return {**TURN_RESET, "question": question, "search_question": search_question,
            "messages": [HumanMessage(message)]}


def finish(state: GraphState) -> dict:
    """Keep only the answer in the conversation: drop this turn's tool calls and results, which are bulky
    and would otherwise be re-sent to the agent on every later turn."""
    messages = state["messages"]
    scratch = [RemoveMessage(id=m.id) for m in messages[turn_start(messages) + 1:]]
    return {"messages": scratch + [AIMessage(state["answer"])]}


def decide_retrieval(state: GraphState) -> dict:
    decision =decider.invoke({"question": state["question"]})
    return {"route": decision.route, "route_reason": decision.reason}


# router, not a node: only reads the state and names the next node
ROUTE_TO_NODE = {"knowledge_base": "plan_queries", "action": "agent", "small_talk": "generate_direct",
                 "distress": "support_wellbeing", "out_of_scope": "decline_out_of_scope"}


def route_after_decision(state: GraphState) -> Literal["plan_queries", "agent", "generate_direct",
                                                       "support_wellbeing", "decline_out_of_scope"]:
    return ROUTE_TO_NODE[state["route"]]


def plan_queries(state: GraphState) -> dict:
    plan = planner.invoke({"question": state["question"]})
    parts = plan.sub_questions[:MAX_SUB_QUESTIONS]
    # one part = no split: search the user's own words, since a rephrased query can lose
    # the exact terms BM25 matches on (e.g. "comp-off" -> "compensatory off")
    return {"sub_questions": parts if len(parts) > 1 else [state.get("search_question") or state["question"]]}


def retrieve(state: GraphState) -> dict:
    # after a rewrite: one new query; otherwise one search per sub-question
    queries = [state["search_query"]] if state.get("search_query") else state["sub_questions"]
    k = TOP_K if len(queries) == 1 else TOP_K_PER_PART
    ranked_lists = [hybrid_search(query, k) for query in queries]
    # round-robin: every part's best chunk first, then every part's second-best, ...
    # (appending list after list would bury part 2's top chunk behind part 1's weakest)
    docs, seen = [], set()
    for rank in range(k):
        for ranked in ranked_lists:
            if rank < len(ranked):
                d = ranked[rank]
                key = (d.metadata["source"], d.page_content)   # the same chunk can come back for two parts
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    docs = state["documents"]
    grades = grader.batch([
        {"question": state["question"], "parts": "; ".join(state["sub_questions"]), "chunk": d.page_content,
         **{k: d.metadata.get(k, "?") for k in ("doc_id", "title", "status", "applies_to")}}
        for d in docs
    ], return_exceptions=True)  # one LLM call per chunk, sent in parallel
    # a chunk whose grade still failed after the retries is kept (the grader is lenient anyway, and
    # check_support catches anything it wrongly lets through) rather than failing the whole question
    grades = [DocGrade(is_relevant=True, reason=f"not graded ({type(g).__name__}), kept")
              if isinstance(g, Exception) else g for g in grades]
    # no reducer on "documents", so this filtered list replaces the one retrieve wrote
    return {
        "documents": [d for d, g in zip(docs, grades) if g.is_relevant],
        # title/department/owner: suggest_contact uses them when nothing answers the question
        "doc_grades": [{"doc_id": d.metadata.get("doc_id"), "relevant": g.is_relevant, "reason": g.reason,
                        **{k: d.metadata.get(k, "?") for k in ("title", "department", "owner")}}
                       for d, g in zip(docs, grades)],
    }


def route_after_grading(state: GraphState) -> Literal["generate", "rewrite_query", "suggest_contact"]:
    if state["documents"]:
        return "generate"
    if state.get("rewrite_count", 0) < MAX_REWRITES:
        return "rewrite_query"
    return "suggest_contact"


def suggest_contact(state: GraphState) -> dict:
    docs = {g["doc_id"]: g for g in state["doc_grades"]}   # one entry per document
    listing = "\n".join(f"- {g['doc_id']}: {g['title']} (team: {g['department']})" for g in docs.values())
    choice = contact_picker.invoke({"question": state["question"], "documents": listing})
    # the LLM only picks a doc_id; the email comes from the document's metadata, so it can't be made up
    contact = docs.get(choice.doc_id)
    tip = (f"The closest policy is {contact['title']} [{contact['doc_id']}], owned by "
           f"{contact['department']}: you can ask them at {contact['owner']}." if contact else
           "Try rephrasing, or raise an OrbitDesk ticket with the relevant team.")
    # an answer was written but didn't help: keep it, add the contact (also when the rewrite then found
    # nothing relevant, so documents is empty but the earlier answer is still in the state)
    if state.get("answer"):
        return {"answer": state["answer"] + "\n\n" + tip}
    return {"answer": "I couldn't find an answer to this in the Orbitra knowledge base. " + tip}


generator = text_chain(prompt, llm)
reviser = text_chain(revise_prompt, llm)
direct_answerer = text_chain(direct_prompt, llm)


def generate(state: GraphState) -> dict:
    answer = generator.invoke({
        "context": format_docs(state["documents"]),
        "question": state["question"],
    })
    return {"answer": answer, "generation_attempts": 1}


def revise_answer(state: GraphState) -> dict:
    claims = "\n".join(f"- {c}" for c in state["unsupported_claims"])
    answer = reviser.invoke({
        "context": format_docs(state["documents"]),
        "question": state["question"],
        "answer": state["answer"],
        "claims": claims,
    })
    return {"answer": answer, "generation_attempts": state["generation_attempts"] + 1}


def check_support(state: GraphState) -> dict:
    grade = support_checker.invoke({
        "context": format_docs(state["documents"]),
        "question": state["question"],  # the user's own numbers are valid inputs to the answer
        "answer": state["answer"],
    })
    # a verdict that names no claim gives revise_answer nothing to fix and flag_unsupported nothing
    # to list, so treat it as supported
    support = grade.support if grade.unsupported_claims else "fully"
    return {"support": support, "unsupported_claims": grade.unsupported_claims}


# END is just the string "__end__", so a router can finish the graph directly
def route_after_support(state: GraphState) -> Literal["check_useful", "revise_answer", "flag_unsupported"]:
    if state["support"] == "fully":
        return "check_useful"
    if state["generation_attempts"] < MAX_GENERATIONS:
        return "revise_answer"
    return "flag_unsupported"


def flag_unsupported(state: GraphState) -> dict:
    claims = "\n".join(f"- {c}" for c in state["unsupported_claims"])
    return {"answer": state["answer"] + "\n\n⚠️ I couldn't verify these parts against the "
                                        "knowledge base:\n" + claims}


def check_useful(state: GraphState) -> dict:
    grade = usefulness_checker.invoke({"question": state["question"], "answer": state["answer"]})
    return {"is_useful": grade.is_useful, "usefulness_reason": grade.reason}


# not useful after the rewrite (e.g. "I don't know"): point the user to a contact instead
def route_after_useful(state: GraphState) -> Literal["finish", "rewrite_query", "suggest_contact"]:
    if state["is_useful"]:
        return "finish"
    if state.get("rewrite_count", 0) < MAX_REWRITES:
        return "rewrite_query"
    return "suggest_contact"


def rewrite_query(state: GraphState) -> dict:
    new_query = rewriter.invoke({
        "question": state["question"],
        "previous_query": state.get("search_query") or state.get("search_question") or state["question"],
    })
    return {"search_query": new_query, "rewrite_count": state.get("rewrite_count", 0) + 1}


def generate_direct(state: GraphState) -> dict:
    return {"answer": direct_answerer.invoke({"question": state["question"]})}


def decline_out_of_scope(state: GraphState) -> dict:
    return {"answer": OUT_OF_SCOPE_REPLY}


def support_wellbeing(state: GraphState) -> dict:
    return {"answer": WELLBEING_REPLY}


def agent(state: GraphState) -> dict:
    messages = state["messages"]
    start = turn_start(messages)
    # system prompt (rebuilt each call, not stored) + recent earlier turns + this turn's question and tool work.
    # The question is sent in its standalone form (with the user's facts) like on the knowledge-base route;
    # the raw message stays in the state, so the conversation history still shows what the user typed.
    history = ([SystemMessage(AGENT_SYSTEM.format(user=describe_user(state.get("employee_id"))))]
               + recent_history(messages[:start]) + [HumanMessage(state["question"])] + messages[start + 1:])
    rounds = state.get("tool_rounds", 0)
    if rounds < MAX_TOOL_ROUNDS:
        response = agent_llm.invoke(history)
    else:   # out of tool rounds: ask for the answer (the nudge is only sent, not kept in the state)
        response = final_llm.invoke(history + [HumanMessage(AGENT_ANSWER_NOW)])
    # tool calls past the limit are ignored, even if the provider doesn't honour tool_choice="none"
    if response.tool_calls and rounds < MAX_TOOL_ROUNDS:
        return {"messages": [response], "tool_rounds": rounds + 1}
    # the model sometimes ends with an empty message after a long search; never return a blank answer
    return {"messages": [response], "answer": response.content.strip() or AGENT_EMPTY_REPLY}


# routes on the answer, not on tool_calls, so a tool call the agent ignored above can't loop back to tools
# (start_turn resets answer to "" each turn)
def route_after_agent(state: GraphState) -> Literal["tools", "finish"]:
    return "finish" if state.get("answer") else "tools"


builder = StateGraph(GraphState)
builder.add_node("start_turn", start_turn)
builder.add_node("finish", finish)
builder.add_node("decide_retrieval", decide_retrieval)
builder.add_node("plan_queries", plan_queries)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("generate", generate)
builder.add_node("check_support", check_support)
builder.add_node("revise_answer", revise_answer)
builder.add_node("flag_unsupported", flag_unsupported)
builder.add_node("check_useful", check_useful)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("suggest_contact", suggest_contact)
builder.add_node("generate_direct", generate_direct)
builder.add_node("decline_out_of_scope", decline_out_of_scope)
builder.add_node("support_wellbeing", support_wellbeing)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(AGENT_TOOLS))
builder.add_edge(START, "start_turn")
builder.add_edge("start_turn", "decide_retrieval")
builder.add_conditional_edges("decide_retrieval", route_after_decision)
builder.add_edge("plan_queries", "retrieve")
builder.add_edge("retrieve", "grade_documents")
builder.add_conditional_edges("grade_documents", route_after_grading)
builder.add_edge("generate", "check_support")
builder.add_conditional_edges("check_support", route_after_support)
builder.add_edge("revise_answer", "check_support")
builder.add_conditional_edges("check_useful", route_after_useful)
builder.add_edge("rewrite_query", "retrieve")  # loop back with the new search query
builder.add_edge("flag_unsupported", "finish")
builder.add_edge("suggest_contact", "finish")
builder.add_edge("generate_direct", "finish")
builder.add_edge("decline_out_of_scope", "finish")
builder.add_edge("support_wellbeing", "finish")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")   # tool results go back to the agent
builder.add_edge("finish", END)
# app: one question per run, no memory (what the evals use); chat_app: remembers each conversation
# (thread_id) in memory until the process exits
app = builder.compile()
chat_app = builder.compile(checkpointer=InMemorySaver())


def ask(question: str, employee_id: str | None = None, thread_id: str | None = None):
    """Answer one message; with a thread_id it is a turn in that conversation."""
    graph, config = (chat_app, {"configurable": {"thread_id": thread_id}}) if thread_id else (app, None)
    for step in graph.stream({"question": question, "employee_id": employee_id}, config, stream_mode="updates"):
        for node, update in step.items():
            if node == "finish":
                continue
            print(f"--- {node} ---")
            if node == "start_turn":
                if update["question"] != question:
                    print(f"  💬 as a standalone question: {update['question']!r}")
            elif node == "decide_retrieval":
                print(f"  route={update['route']}  ({update['route_reason']})")
            elif node == "plan_queries":
                print(f"  🧩 parts: {update['sub_questions']}")
            elif node == "retrieve":
                for d in update["documents"]:
                    print(f"  [{d.metadata.get('doc_id')}] status={d.metadata.get('status')}  "
                          f"{d.page_content[:80]!r}")
            elif node == "grade_documents":
                for g in update["doc_grades"]:
                    print(f"  {'✓' if g['relevant'] else '✗'} [{g['doc_id']}] {g['reason']}")
            elif node == "check_support":
                print(f"  support={update['support']}")
                for c in update["unsupported_claims"]:
                    print(f"  ✗ {c}")
            elif node == "check_useful":
                print(f"  useful={update['is_useful']}  ({update['usefulness_reason']})")
            elif node == "rewrite_query":
                print(f"  🔁 new query: {update['search_query']!r}")
            elif node == "tools":
                for m in update["messages"]:
                    print(f"  ⇐ {m.name}: {m.content[:150]!r}")
            elif node == "agent" and "answer" not in update:
                for call in update["messages"][-1].tool_calls:
                    print(f"  🔧 {call['name']}({call['args']})")
            else:
                print(update["answer"])


def main():
    parser = argparse.ArgumentParser(description="Chat with OrbitBot.")
    parser.add_argument("--employee-id", help="sign in as this employee (e.g. ORB01001) for personal data "
                                              "such as leave balances")
    args = parser.parse_args()
    if args.employee_id and args.employee_id not in agent_tools.EMPLOYEES_BY_ID:
        parser.error(f"no employee {args.employee_id!r} in data/employee_directory.csv")
    if args.employee_id:
        print(f"Signed in as {agent_tools.EMPLOYEES_BY_ID[args.employee_id]['name']} ({args.employee_id}).")
    print("OrbitBot is ready. Ask a question, type 'new' to start a new conversation, or 'exit' to quit.")
    thread_id = str(uuid.uuid4())
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "new":
            thread_id = str(uuid.uuid4())
            print("Started a new conversation.")
        elif question:
            ask_safely(question, args.employee_id, thread_id)
    print("\nBye!")


def ask_safely(question: str, employee_id: str | None, thread_id: str):
    """ask(), but a failed turn (an LLM error that outlasted the retries, a 429, ...) doesn't end the chat.
    The failed turn's messages are removed from the conversation, so the next turn starts clean: a half-done
    turn could leave the user's message without a reply, or an agent tool call without its result, which the
    LLM API rejects on every later turn."""
    config = {"configurable": {"thread_id": thread_id}}
    before = {m.id for m in chat_app.get_state(config).values.get("messages", [])}
    try:
        ask(question, employee_id, thread_id)
    except Exception as e:
        print(f"\n⚠️ Sorry, something went wrong ({type(e).__name__}: {str(e)[:200]}). "
              "Please try again; the conversation so far is kept.")
        added = [RemoveMessage(id=m.id) for m in chat_app.get_state(config).values.get("messages", [])
                 if m.id not in before]
        if added:   # as_node="finish": the thread then counts as a finished turn
            chat_app.update_state(config, {"messages": added}, as_node="finish")


if __name__ == "__main__":
    main()
