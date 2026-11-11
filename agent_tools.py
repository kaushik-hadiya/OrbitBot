"""Agent tools for OrbitBot's `action` route, backed by the mock data in data/.

Everything here is read-only or a draft: no tool changes anything outside this process, so the bot can
never claim to have done something it didn't (see the fabricated_action cases in eval/safety_questions.jsonl).
Submitting a ticket needs a human-approval step first, so for now the bot only drafts one.
"""
import ast
import csv
import operator
import random
import re
from datetime import date
from pathlib import Path
from typing import Annotated

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

DATA_DIR = Path("data")
# the knowledge base is written as of this date (KB_README.md), so "today" is fixed for reproducible evals
TODAY = date(2026, 10, 19)

INDIA = {"Bengaluru", "Pune", "Hyderabad"}

with open(DATA_DIR / "employee_directory.csv", encoding="utf-8") as f:
    EMPLOYEES = list(csv.DictReader(f))
EMPLOYEES_BY_ID = {e["employee_id"]: e for e in EMPLOYEES}

with open(DATA_DIR / "ticket_history_2026.csv", encoding="utf-8") as f:
    TICKETS = list(csv.DictReader(f))

# directory fields anyone at Orbitra can see; level and joining date stay private
PUBLIC_FIELDS = ("employee_id", "name", "email", "department", "title", "location", "slack_handle", "work_mode")
MAX_MATCHES = 5


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# requester_id is left out of the searchable text and the metadata: similar tickets must not reveal who asked
_ticket_index = BM25Retriever.from_documents(
    [Document(page_content=f"{t['subcategory']}. {t['subject']}. {t['description']}",
              metadata={k: v for k, v in t.items() if not k.startswith("requester_")})
     for t in TICKETS if t["resolution"]],   # open tickets have no resolution to learn from
    k=15, preprocess_func=_tokenize,
)
MAX_SIMILAR = 3


@tool
def get_current_date() -> str:
    """Today's date and weekday. Use it for anything that depends on today (deadlines, 'next' events, freezes)."""
    return TODAY.strftime("%A %d %B %Y (%Y-%m-%d)")


@tool
def lookup_employee(query: str) -> str:
    """Find colleagues in the Orbitra employee directory by name, email, employee ID or Slack handle.
    Returns only public directory fields: name, email, department, title, office location, Slack, work mode.
    It has no salaries, phone numbers, addresses, managers or performance data."""
    q = query.strip().lower().lstrip("@")
    if not q:
        return "Give a name, email, employee ID or Slack handle to look up."
    matches = [e for e in EMPLOYEES
               if q in (e["employee_id"].lower(), e["email"].lower())
               or q in e["name"].lower() or q in e["slack_handle"].lower()]
    if not matches:
        return f"No one in the directory matches {query!r}."
    lines = [", ".join(f"{k}: {e[k]}" for k in PUBLIC_FIELDS) for e in matches[:MAX_MATCHES]]
    more = f"\n({len(matches) - MAX_MATCHES} more matches; ask for a fuller name.)" if len(matches) > MAX_MATCHES else ""
    return "\n".join(lines) + more


def _months_credited(joined: date) -> int:
    """India EL is credited on the 1st of each month: count the 1sts from the later of joining and 1 Jan up to today."""
    start = max(joined, date(TODAY.year, 1, 1))
    first = start.month if start.day == 1 else start.month + 1
    return max(0, TODAY.month - first + 1)


def _pro_rated(days: float, joined: date) -> float:
    """Front-loaded leave is pro-rated by month for this year's joiners (rounded down to half days)."""
    if joined.year < TODAY.year:
        return days
    return int(days * (13 - joined.month) / 12 * 2) / 2


def _mock_balances(emp: dict) -> list[tuple[str, float, float]]:
    """(leave type, available this year, used) from the leave policies, with random but stable usage per employee."""
    rng = random.Random(emp["employee_id"])
    joined = date.fromisoformat(emp["date_of_joining"])
    years = TODAY.year - joined.year - ((TODAY.month, TODAY.day) < (joined.month, joined.day))
    if emp["location"] in INDIA:   # HR-LV-001 v5
        carried = rng.randint(0, 12) if joined.year < TODAY.year else 0
        earned = 1.5 * _months_credited(joined)
        types = [("Earned Leave (EL)", min(carried + earned, 30)),
                 ("Sick Leave (SL)", _pro_rated(12, joined)), ("Casual Leave (CL)", _pro_rated(6, joined))]
    elif emp["location"] == "Austin":   # HR-LV-002: PTO accrues over 24 semi-monthly periods
        annual = 20 if years <= 2 else 23 if years <= 5 else 25
        periods = (TODAY.month - 1) * 2 + (2 if TODAY.day >= 15 else 1)
        types = [("PTO", round(annual * periods / 24, 1)), ("Sick days", _pro_rated(10, joined))]
    else:   # London, HR-LV-003: 25 days + 1 per completed year, max 30
        types = [("Annual leave", _pro_rated(min(25 + years, 30), joined))]
    return [(name, total, rng.randint(0, int(total * 0.8) * 2) / 2) for name, total in types]


@tool
def get_leave_balance(state: Annotated[dict, InjectedState]) -> str:
    """The signed-in employee's own leave balances, from PeopleHub. Takes no arguments: it can only show
    the balances of the person using OrbitBot, never anyone else's."""
    # the employee ID comes from the graph state, not from the model, so it can't be asked for someone else
    emp = EMPLOYEES_BY_ID.get(state.get("employee_id") or "")
    if not emp:
        return ("Not signed in, so balances are unavailable. The employee can see them in PeopleHub → Leave → "
                "My Balances.")
    rows = [f"- {name}: {total - used:g} days left ({used:g} used of {total:g} available so far this year)"
            for name, total, used in _mock_balances(emp)]
    return (f"Leave balances for {emp['name']} ({emp['location']}) as of {TODAY:%d %b %Y}:\n" + "\n".join(rows)
            + "\nIf a balance looks wrong, raise OrbitDesk → HR → 'Leave balance correction'.")


@tool
def search_similar_tickets(description: str) -> str:
    """Search past OrbitDesk tickets (2026) for issues like the user's, to see how they were resolved and
    which knowledge-base article covers them. Use it for 'X is broken / not working' problems."""
    # many tickets repeat the same issue word for word: keep one per subject + resolution
    unique = {}
    for t in (h.metadata for h in _ticket_index.invoke(description)):
        unique.setdefault((t["subject"], t["resolution"]), t)
    if not unique:
        return "No similar past tickets."
    return "\n".join(
        f"- {t['ticket_id']} [{t['category']} → {t['subcategory']}, {t['priority']}] {t['subject']}. "
        f"Resolution: {t['resolution']} (took {t['resolution_time_hours']} h; KB article: {t['kb_article'] or 'none'})"
        for t in list(unique.values())[:MAX_SIMILAR])


TICKET_CATEGORIES = sorted({t["category"] for t in TICKETS})
SUBCATEGORIES = {c: sorted({t["subcategory"] for t in TICKETS if t["category"] == c}) for c in TICKET_CATEGORIES}


@tool
def draft_ticket(category: str, subcategory: str, subject: str, description: str, priority: str = "P4") -> str:
    """Prepare an OrbitDesk ticket for the user to submit themselves. It does NOT submit anything.
    category: one of IT, HR, Payroll, Benefits, Finance, Admin, Legal, Security, Learning.
    subcategory: an existing OrbitDesk subcategory, e.g. IT: Laptop issue, MFA reset, Password reset,
    Hardware request, VPN issue; HR: Leave balance, Letters; Finance: Expense query (an unknown one is rejected
    with the valid list).
    priority: P1 (blocked, can't work) to P4 (routine); default P4."""
    cat = next((c for c in TICKET_CATEGORIES if c.lower() == category.strip().lower()), None)
    if not cat:
        return f"Unknown category {category!r}. Use one of: {', '.join(TICKET_CATEGORIES)}."
    if priority not in {"P1", "P2", "P3", "P4"}:
        priority = "P4"
    known = SUBCATEGORIES[cat]
    sub = next((s for s in known if s.lower() == subcategory.strip().lower()), None)
    if not sub:   # a made-up subcategory would send the ticket to the wrong queue
        return f"Unknown {cat} subcategory {subcategory!r}. Use one of: {', '.join(known)}."
    return (f"DRAFT (not submitted) — OrbitDesk → {cat} → {sub}, priority {priority}\n"
            f"Subject: {subject}\nDescription: {description}\n"
            f"The user must submit it at help.orbitra.internal or with /orbitdesk in Slack.")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers, + - * / // % ** and brackets are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '45000 / 30 * 12' for leave encashment or per-diem maths.
    Numbers, + - * / // % ** and brackets only."""
    try:
        result = _eval(ast.parse(expression.replace(",", ""), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as e:
        return f"Can't calculate {expression!r}: {e}"
    result = round(result, 4)
    return f"{expression} = {int(result) if result == int(result) else result}"


TOOLS = [get_current_date, lookup_employee, get_leave_balance, search_similar_tickets, draft_ticket, calculator]
