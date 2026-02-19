# Building a Multi-Agent Badminton Scheduler: A Deep Dive into Google's Agent2Agent (A2A) Protocol

*Published on Medium — #AI #Agents #GoogleADK #A2A #MultiAgent*

---

> **TL;DR:** I built a real-world multi-agent system where an AI "Host" agent coordinates with two independently-built AI "Friend" agents—each running on different frameworks (Google ADK and CrewAI)—to find a common time slot and book a badminton court. The secret glue? Google's open **Agent2Agent (A2A) Protocol**. Here's everything I learned.

---

## 🏸 The Problem: Scheduling is Hard. Let's Make Agents Do It.

Picture this: you want to play badminton this weekend. You pull out your phone and instead of texting three different friends, asking them for their availability, waiting for replies, cross-referencing everyone's schedule with court availability, and booking — *what if a network of AI agents did all of that for you automatically?*

That's exactly what I built. But more importantly, this project is a **hands-on demonstration of A2A (Agent2Agent) Protocol** — Google's open standard for letting AI agents talk to each other, regardless of which framework they're built with.

Let me walk you through every concept and every line of code.

---

## 🤔 What Is the A2A Protocol?

The **Agent2Agent (A2A) Protocol** is an open standard developed by Google that defines how AI agents — even those built by completely different teams using completely different frameworks — can discover each other, communicate, and collaborate on tasks.

Think of it like HTTP for human browsers, but **HTTP for AI agents**.

Before A2A existed, if you wanted agents to work together, you had to build custom, brittle, one-off integrations. A2A solves this by defining a **common language** for:

- 🔍 **Agent Discovery** — An `AgentCard` (a JSON metadata file) that describes what an agent can do
- 📨 **Message Sending** — Standardized `SendMessageRequest / SendMessageResponse` types
- 📋 **Task Lifecycle** — Tasks have states: `submitted → working → completed`
- 📦 **Artifact Exchange** — Agents return results as `Artifacts` containing `Parts` (text, files, etc.)

The protocol is transport-agnostic (uses HTTP/HTTPS) and uses **JSON-RPC** under the hood, making it easy to integrate with any stack.

---

## 🏗️ System Architecture

Our badminton scheduling system has three agents:

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (via ADK UI)                        │
│             "Schedule a badminton game this week!"               │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST / ADK Dev UI
                             ▼
        ┌────────────────────────────────────────┐
        │         HOST AGENT (Google ADK)         │
        │            Port: 10001 (ADK)             │
        │  Tools:                                  │
        │  • send_message()       ← A2A client     │
        │  • list_court_availabilities()           │
        │  • book_badminton_court()                │
        └──────────────┬────────────┬─────────────┘
                       │            │
          A2A Protocol │            │ A2A Protocol
          (HTTP POST)  │            │ (HTTP POST)
                       ▼            ▼
     ┌──────────────────┐  ┌─────────────────────┐
     │  SHUBHAM AGENT   │  │   SUSHMITA AGENT     │
     │  (Google ADK)    │  │   (CrewAI)           │
     │  Port: 10002     │  │   Port: 10003        │
     │  Tool:           │  │   Tool:              │
     │  get_availability│  │   AvailabilityTool   │
     └──────────────────┘  └─────────────────────┘
```

**The key insight:** The Host Agent doesn't care *at all* that Shubham uses Google ADK while Sushmita uses CrewAI. It just speaks A2A to both. That's the power of the protocol.

---

## 📁 Project Structure

```
A2A-Agent2Agent-Protocol-demo/
│
├── host_agent/                  # The orchestrating agent
│   └── host/
│       ├── agent.py             # HostAgent class (ADK-based)
│       ├── badminton_tools.py   # Court booking tools
│       └── remote_agent_connection.py  # A2A client wrapper
│
├── shubham_agent/               # Friend #1 (Google ADK)
│   ├── agent.py                 # LlmAgent + get_availability tool
│   ├── agent_executor.py        # A2A server executor bridge
│   └── __main__.py              # HTTP server entrypoint
│
└── sushmita_agent/              # Friend #2 (CrewAI)
    ├── agent.py                 # CrewAI SchedulingAgent
    ├── agent_executor.py        # A2A server executor bridge
    └── __main__.py              # HTTP server entrypoint
```

---

## 🔄 How the A2A Protocol Works: Step by Step

Here's the complete end-to-end flow when a user says *"Schedule badminton this week!"*:

```
User Input
  │
  ▼
Host Agent receives query via ADK Runner
  │
  ├─ 1. Calls send_message("Shubham Agent", "Are you free this week?")
  │       │
  │       ├─ Builds SendMessageRequest with task_id, context_id, message parts
  │       ├─ HTTP POST to http://localhost:10002/
  │       ├─ Shubham's A2A server receives it
  │       ├─ ShubhamAgentExecutor.execute() is called
  │       │       └─ ADK Runner runs LlmAgent with get_availability() tool
  │       └─ Returns Task with Artifact (available time slots as text)
  │
  ├─ 2. Calls send_message("Sushmita Agent", "Are you free this week?")
  │       │
  │       ├─ HTTP POST to http://localhost:10003/
  │       ├─ Sushmita's A2A server receives it
  │       ├─ SchedulingAgentExecutor.execute() is called
  │       │       └─ CrewAI Crew.kickoff() runs AvailabilityTool
  │       └─ Returns Task with Artifact (available time slots as text)
  │
  ├─ 3. Host Agent LLM analyzes both responses → finds common slots
  ├─ 4. Calls list_court_availabilities(date) to check court
  ├─ 5. Presents options to user, user confirms
  └─ 6. Calls book_badminton_court(date, start, end, name)
         └─ Returns booking confirmation with ID
```

---

## 🧠 Deep Dive: The Host Agent

The Host Agent is the **orchestrator**. It's built using Google's Agent Development Kit (ADK) and acts as the intelligent coordinator in the system.

### Setting Up All Dependencies

```python
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard, MessageSendParams, SendMessageRequest,
    SendMessageResponse, SendMessageSuccessResponse, Task,
)
from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
```

This imports from **both** worlds: the `a2a` package (the protocol library) and `google.adk` (Google's agent framework). The `A2ACardResolver` is used to discover remote agents, while `A2AClient` handles the actual message sending.

### The HostAgent Class

```python
class HostAgent:
    """The Host agent."""

    def __init__(self):
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "host_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
```

**Key design decisions here:**
- `remote_agent_connections` is a dictionary mapping an agent's **name** to its connection object. Once the agent card is fetched, we refer to friends **by name** (e.g., `"Shubham Agent"`).
- `InMemoryArtifactService`, `InMemorySessionService`, and `InMemoryMemoryService` are lightweight, non-persistent services — perfect for demos.
- `Runner` is the ADK execution engine. It handles the conversation loop, tool calling, and state management.

### Agent Discovery: Fetching AgentCards

One of the most elegant features of A2A is **agent discovery**. Before the host can talk to any friend, it asks each one *"who are you and what can you do?"* This is done via the `/agent` endpoint that the A2A server automatically exposes.

```python
async def _async_init_components(self, remote_agent_addresses: List[str]):
    async with httpx.AsyncClient(timeout=30) as client:
        for address in remote_agent_addresses:
            card_resolver = A2ACardResolver(client, address)
            try:
                card = await card_resolver.get_agent_card()
                remote_connection = RemoteAgentConnections(
                    agent_card=card, agent_url=address
                )
                self.remote_agent_connections[card.name] = remote_connection
                self.cards[card.name] = card
            except httpx.ConnectError as e:
                print(f"ERROR: Failed to get agent card from {address}: {e}")

    agent_info = [
        json.dumps({"name": card.name, "description": card.description})
        for card in self.cards.values()
    ]
    self.agents = "\n".join(agent_info) if agent_info else "No friends found"
```

`A2ACardResolver.get_agent_card()` makes an HTTP GET to `{address}/.well-known/agent.json`. The returned `AgentCard` contains the agent's name, description, skills, supported capabilities, and URL. This is then **injected into the Host Agent's system prompt** dynamically:

```python
def root_instruction(self, context: ReadonlyContext) -> str:
    return f"""
    **Role:** You are the Host Agent, an expert scheduler for badminton games.
    ...
    <Available Agents>
    {self.agents}
    </Available Agents>
    """
```

This is beautifully dynamic — if a new friend agent comes online, the Host Agent automatically knows about it without any code changes!

### The `send_message` Tool: A2A in Action

This is the heart of the A2A protocol interaction — the actual tool that the LLM calls when it wants to talk to a friend agent:

```python
async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
    """Sends a task to a remote friend agent."""
    if agent_name not in self.remote_agent_connections:
        raise ValueError(f"Agent {agent_name} not found")
    client = self.remote_agent_connections[agent_name]

    # Every conversation needs unique IDs for tracking
    state = tool_context.state
    task_id = state.get("task_id", str(uuid.uuid4()))
    context_id = state.get("context_id", str(uuid.uuid4()))
    message_id = str(uuid.uuid4())

    payload = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": task}],
            "messageId": message_id,
            "taskId": task_id,
            "contextId": context_id,
        },
    }

    message_request = SendMessageRequest(
        id=message_id, params=MessageSendParams.model_validate(payload)
    )
    send_response: SendMessageResponse = await client.send_message(message_request)
```

**Breaking this down:**

1. **`task_id`** — Uniquely identifies this specific task (e.g., "check Shubham's availability")
2. **`context_id`** — Groups multiple related messages into a single conversation context
3. **`message_id`** — Uniquely identifies this specific message within the task
4. **`MessageSendParams`** — Pydantic model that validates the payload structure

The `SendMessageRequest` is then sent via HTTP POST to the remote agent. The response is a `SendMessageResponse` that wraps either a `SendMessageSuccessResponse` (containing a `Task` with `Artifacts`) or an error.

```python
    if not isinstance(send_response.root, SendMessageSuccessResponse) \
       or not isinstance(send_response.root.result, Task):
        print("Received a non-success or non-task response. Cannot proceed.")
        return

    response_content = send_response.root.model_dump_json(exclude_none=True)
    json_content = json.loads(response_content)

    resp = []
    if json_content.get("result", {}).get("artifacts"):
        for artifact in json_content["result"]["artifacts"]:
            if artifact.get("parts"):
                resp.extend(artifact["parts"])
    return resp
```

The response parsing walks through the `Task` → `artifacts` → `parts` chain to extract the text content from the friend agent's response.

### Streaming Responses

The Host Agent supports **streaming** — it yields intermediate "thinking" updates while the final response is being generated:

```python
async def stream(self, query: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
    async for event in self._runner.run_async(
        user_id=self._user_id, session_id=session.id, new_message=content
    ):
        if event.is_final_response():
            yield {
                "is_task_complete": True,
                "content": response,
            }
        else:
            yield {
                "is_task_complete": False,
                "updates": "The host agent is thinking...",
            }
```

This lets the frontend display real-time progress, which is critical for UX when agents are doing multi-step reasoning.

---

## 🏓 Deep Dive: Shubham Agent (Google ADK)

Shubham's agent is built entirely with **Google ADK** (`LlmAgent`). It's the simpler of the two friend agents.

### The Core Agent Logic

```python
import random
from datetime import date, datetime, timedelta
from google.adk.agents import LlmAgent

def generate_shubham_calendar() -> dict[str, list[str]]:
    """Generates a random calendar for Shubham for the next 7 days."""
    calendar = {}
    today = date.today()
    possible_times = [f"{h:02}:00" for h in range(8, 21)]  # 8 AM to 8 PM

    for i in range(7):
        current_date = today + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        # 8 random unique time slots out of 13 possible
        available_slots = sorted(random.sample(possible_times, 8))
        calendar[date_str] = available_slots

    return calendar

KARLEY_CALENDAR = generate_shubham_calendar()
```

The calendar is **generated at module load time** (when the server starts), making it stable for the duration of the agent's lifecycle. The agent uses random sampling to simulate a realistic partial availability.

### The Availability Tool

```python
def get_availability(start_date: str, end_date: str) -> str:
    """
    Checks Shubham's availability for a given date range.

    Args:
        start_date: The start of the date range to check, in YYYY-MM-DD format.
        end_date: The end of the date range to check, in YYYY-MM-DD format.

    Returns:
        A string listing Shubham's available times for that date range.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        results = []
        delta = end - start
        for i in range(delta.days + 1):
            day = start + timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            available_slots = KARLEY_CALENDAR.get(date_str, [])
            if available_slots:
                availability = f"On {date_str}, Shubham is available at: {', '.join(available_slots)}."
                results.append(availability)
            else:
                results.append(f"Shubham is not available on {date_str}.")

        return "\n".join(results)
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD for both start and end dates."
```

Notice: this is a **plain Python function**. Google ADK automatically introspects function signatures and docstrings to create tool schemas. No decorators needed! The LLM understands what the tool does from the docstring and type hints.

### Creating the ADK Agent

```python
def create_agent() -> LlmAgent:
    """Constructs the ADK agent for Shubham."""
    return LlmAgent(
        model="gemini-2.0-flash",
        name="Shubham_Agent",
        instruction="""
            **Role:** You are Shubham's personal scheduling assistant.
            Your sole responsibility is to manage her calendar and respond
            to inquiries about her availability for badminton.

            **Core Directives:**
            *   **Check Availability:** Use the `get_shubham_availability`
                tool to determine if Shubham is free on a requested date range.
            *   **Polite and Concise:** Always be polite and to the point.
            *   **Stick to Your Role:** Do not engage in any conversation
                outside of scheduling.
        """,
        tools=[get_availability],
    )
```

The `LlmAgent` is the workhorse — it takes:
- `model`: Which Gemini model to use for reasoning
- `name`: How this agent identifies itself
- `instruction`: The system prompt that defines behavior
- `tools`: List of Python functions the LLM can call

### The AgentExecutor: Bridging ADK and A2A

This is where the magic happens — the **AgentExecutor** is the bridge between the A2A protocol server and the actual ADK agent logic:

```python
class ShubhamAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs Shubham's ADK-based Agent."""

    def __init__(self, runner: Runner):
        self.runner = runner

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()

        await self._process_request(
            types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts),
            ),
            context.context_id,
            updater,
        )
```

**Key concepts:**

- **`RequestContext`** — Contains the incoming A2A message, task ID, context ID, and current task state
- **`EventQueue`** — The channel through which the executor pushes status updates and results back to the A2A server
- **`TaskUpdater`** — A helper that wraps the event queue with convenient methods: `submit()`, `start_work()`, `add_artifact()`, `complete()`

The lifecycle:
1. Task arrives → `updater.submit()` transitions state to `submitted`
2. Processing begins → `updater.start_work()` transitions to `working`
3. Result ready → `updater.add_artifact(parts)` + `updater.complete()` → `completed`

### The Part Conversion Functions

A crucial detail: A2A uses its own `Part` types (protocol-level), while Google ADK uses `google.genai.types.Part`. You need bidirectional conversion:

```python
def convert_a2a_parts_to_genai(parts: list[Part]) -> list[types.Part]:
    """Convert A2A Part types into Google Gen AI Part types."""
    return [convert_a2a_part_to_genai(part) for part in parts]

def convert_a2a_part_to_genai(part: Part) -> types.Part:
    root = part.root
    if isinstance(root, TextPart):
        return types.Part(text=root.text)
    if isinstance(root, FilePart):
        if isinstance(root.file, FileWithUri):
            return types.Part(
                file_data=types.FileData(
                    file_uri=root.file.uri,
                    mime_type=root.file.mimeType
                )
            )
        if isinstance(root.file, FileWithBytes):
            return types.Part(
                inline_data=types.Blob(
                    data=root.file.bytes.encode("utf-8"),
                    mime_type=root.file.mimeType or "application/octet-stream",
                )
            )
    raise ValueError(f"Unsupported part type: {type(part)}")
```

This is the protocol boundary — incoming A2A messages get deserialized into ADK types for the LLM, and outgoing ADK responses get serialized back into A2A types for the protocol.

### Booting the A2A Server

```python
def main():
    host = "localhost"
    port = 10002

    capabilities = AgentCapabilities(streaming=True)
    skill = AgentSkill(
        id="check_schedule",
        name="Check Shubham's Schedule",
        description="Checks Shubham's availability for a badminton game.",
        tags=["scheduling", "calendar"],
        examples=["Is Shubham free to play badminton tomorrow?"],
    )
    agent_card = AgentCard(
        name="Shubham Agent",
        description="An agent that manages Shubham's schedule for badminton games.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        capabilities=capabilities,
        skills=[skill],
    )

    adk_agent = create_agent()
    runner = Runner(
        app_name=agent_card.name,
        agent=adk_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    agent_executor = ShubhamAgentExecutor(runner)

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    uvicorn.run(server.build(), host=host, port=port)
```

Every piece here serves a purpose:

| Component | Role |
|-----------|------|
| `AgentCard` | The agent's "business card" — advertised at `/.well-known/agent.json` |
| `AgentSkill` | Describes a capability (searchable, filterable) |
| `AgentCapabilities` | Declares what the agent supports (`streaming=True`) |
| `Runner` | ADK execution engine with all services |
| `ShubhamAgentExecutor` | Bridge between A2A protocol and ADK |
| `DefaultRequestHandler` | Routes incoming A2A HTTP requests |
| `InMemoryTaskStore` | Stores active task state in RAM |
| `A2AStarletteApplication` | ASGI application (Starlette/FastAPI-compatible) |
| `uvicorn.run()` | Lightweight ASGI server |

---

## 🤖 Deep Dive: Sushmita Agent (CrewAI) — A Different Framework!

Here's where things get **really interesting**. Sushmita's agent is built with **CrewAI** — a completely different framework than Google ADK. Yet it slots into the exact same A2A protocol infrastructure.

This is the whole point of A2A: **framework agnosticism**.

### The CrewAI Agent Setup

```python
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class AvailabilityToolInput(BaseModel):
    """Input schema for AvailabilityTool."""
    date_range: str = Field(
        ...,
        description="The date or date range to check for availability, "
                    "e.g., '2024-07-28' or '2024-07-28 to 2024-07-30'.",
    )

class AvailabilityTool(BaseTool):
    name: str = "Calendar Availability Checker"
    description: str = (
        "Checks my availability for a given date or date range. "
        "Use this to find out when I am free."
    )
    args_schema: Type[BaseModel] = AvailabilityToolInput

    def _run(self, date_range: str) -> str:
        """Checks my availability for a given date range."""
        dates_to_check = [d.strip() for d in date_range.split("to")]
        start_date_str = dates_to_check[0]
        end_date_str = dates_to_check[-1]
        # ... lookup MY_CALENDAR ...
```

Notice the difference from the ADK approach:

- **ADK**: Tools are plain Python functions — the framework infers the schema from type hints and docstrings
- **CrewAI**: Tools are classes extending `BaseTool`, with a Pydantic `args_schema` for explicit input validation

Both are valid architectures; they just use different APIs.

### CrewAI's Agent-Task-Crew Pattern

```python
class SchedulingAgent:
    def __init__(self):
        self.llm = LLM(
            model="gemini/gemini-2.0-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

        self.scheduling_assistant = Agent(
            role="Personal Scheduling Assistant",
            goal="Check my calendar and answer questions about my availability.",
            backstory=(
                "You are a highly efficient and polite assistant. Your only job is "
                "to manage my calendar. You are an expert at using the "
                "Calendar Availability Checker tool to find out when I am free."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[AvailabilityTool()],
            llm=self.llm,
        )

    def invoke(self, question: str) -> str:
        """Kicks off the crew to answer a scheduling question."""
        task_description = (
            f"Answer the user's question about my availability. "
            f"The user asked: '{question}'. "
            f"Today's date is {date.today().strftime('%Y-%m-%d')}."
        )

        check_availability_task = Task(
            description=task_description,
            expected_output="A polite and concise answer to the user's availability question.",
            agent=self.scheduling_assistant,
        )

        crew = Crew(
            agents=[self.scheduling_assistant],
            tasks=[check_availability_task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        return str(result)
```

**CrewAI's paradigm:**
- **`Agent`** = A role-playing entity with a goal, backstory, and tools
- **`Task`** = A specific piece of work with a description and expected output
- **`Crew`** = A collection of agents working on tasks (supports sequential or hierarchical processing)
- **`Process.sequential`** = Tasks are executed one by one in order

### Sushmita's Simpler Executor

Because CrewAI's `SchedulingAgent.invoke()` is synchronous and straightforward, the executor is much simpler:

```python
class SchedulingAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = SchedulingAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        query = context.get_user_input()
        try:
            result = self.agent.invoke(query)
        except Exception as e:
            raise ServerError(error=InternalError()) from e

        parts = [Part(root=TextPart(text=result))]
        await updater.add_artifact(parts)
        await updater.complete()
```

Notice `context.get_user_input()` — a convenience method that extracts the plain text from the A2A message parts. No manual conversion needed here since we're just passing a string to CrewAI.

**Important difference:** Sushmita's agent declares `streaming=False` in its `AgentCapabilities` — CrewAI's `Crew.kickoff()` is synchronous and blocking, so streaming isn't supported here.

---

## 🎾 The Badminton Court Tools

The Host Agent has its own domain-specific tools — managing the court booking system.

### Court Schedule Generation

```python
COURT_SCHEDULE: Dict[str, Dict[str, str]] = {}

def generate_court_schedule():
    """Generates a schedule for the badminton court for the next 7 days."""
    global COURT_SCHEDULE
    today = date.today()
    possible_times = [f"{h:02}:00" for h in range(8, 21)]  # 8 AM to 8 PM

    for i in range(7):
        current_date = today + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        COURT_SCHEDULE[date_str] = {time: "unknown" for time in possible_times}

generate_court_schedule()
```

The court schedule is an in-memory dictionary:
```python
{
  "2025-02-20": {
    "08:00": "unknown",  # "unknown" means available
    "09:00": "TeamAlpha",  # booked!
    "10:00": "unknown",
    ...
  }
}
```

### Listing Available Slots

```python
def list_court_availabilities(date: str) -> dict:
    try:
        datetime.strptime(date, "%Y-%m-%d")  # Validate format
    except ValueError:
        return {"status": "error", "message": "Invalid date format."}

    daily_schedule = COURT_SCHEDULE.get(date)
    if not daily_schedule:
        return {"status": "success", "message": f"The court is not open on {date}.", "schedule": {}}

    available_slots = [
        time for time, party in daily_schedule.items() if party == "unknown"
    ]
    booked_slots = {
        time: party for time, party in daily_schedule.items() if party != "unknown"
    }

    return {
        "status": "success",
        "message": f"Schedule for {date}.",
        "available_slots": available_slots,
        "booked_slots": booked_slots,
    }
```

### Booking the Court

```python
def book_badminton_court(
    date: str, start_time: str, end_time: str, reservation_name: str
) -> dict:
    # Parse and validate times
    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")

    if start_dt >= end_dt:
        return {"status": "error", "message": "Start time must be before end time."}

    # Generate all hourly slots in the range
    required_slots = []
    current_time = start_dt
    while current_time < end_dt:
        required_slots.append(current_time.strftime("%H:%M"))
        current_time += timedelta(hours=1)

    # Check for conflicts
    daily_schedule = COURT_SCHEDULE.get(date, {})
    for slot in required_slots:
        if daily_schedule.get(slot, "booked") != "unknown":
            party = daily_schedule.get(slot)
            return {
                "status": "error",
                "message": f"The time slot {slot} on {date} is already booked by {party}.",
            }

    # Commit the booking
    for slot in required_slots:
        COURT_SCHEDULE[date][slot] = reservation_name

    return {
        "status": "success",
        "message": f"Success! The badminton court has been booked for "
                   f"{reservation_name} from {start_time} to {end_time} on {date}.",
    }
```

**The booking logic is elegant:**
1. Convert start/end times into hourly slots (e.g., 10:00 to 12:00 → `["10:00", "11:00"]`)
2. Check each slot for conflicts
3. If all clear, mark all slots with the reservation name

---

## 🌍 The A2A Protocol: Key Concepts Summary

Let's crystallize the core concepts:

### 1. AgentCard — The Agent's Identity

```json
{
  "name": "Shubham Agent",
  "description": "An agent that manages Shubham's schedule for badminton games.",
  "url": "http://localhost:10002/",
  "version": "1.0.0",
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "capabilities": { "streaming": true },
  "skills": [
    {
      "id": "check_schedule",
      "name": "Check Shubham's Schedule",
      "description": "Checks Shubham's availability for a badminton game.",
      "tags": ["scheduling", "calendar"],
      "examples": ["Is Shubham free to play badminton tomorrow?"]
    }
  ]
}
```

This card is automatically served at `http://localhost:10002/.well-known/agent.json` by the A2AStarletteApplication.

### 2. Message Structure — How Agents Talk

```json
{
  "message": {
    "role": "user",
    "parts": [
      { "type": "text", "text": "Are you available for badminton between 2025-02-20 and 2025-02-22?" }
    ],
    "messageId": "550e8400-e29b-41d4-a716-446655440000",
    "taskId": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "contextId": "6ba7b811-9dad-11d1-80b4-00c04fd430c8"
  }
}
```

### 3. Task Lifecycle

```
submitted ──► working ──► completed
                │
                └──► failed (on error)
```

- **submitted**: Task received, queued for processing
- **working**: Agent is actively processing (tool calls may be happening)
- **completed**: Final result ready in `artifacts`

### 4. Response with Artifacts

```json
{
  "result": {
    "id": "task-uuid",
    "status": { "state": "completed" },
    "artifacts": [
      {
        "name": "response",
        "parts": [
          {
            "type": "text",
            "text": "On 2025-02-20, Shubham is available at: 08:00, 10:00, 13:00, ..."
          }
        ]
      }
    ]
  }
}
```

---

## 🔧 Dependencies and Tech Stack

```toml
[project]
name = "a2a-friend-scheduling"
dependencies = [
    "google-adk>=1.2.1",      # Google's Agent Development Kit
    "a2a-sdk>=0.2.5",         # A2A Protocol SDK
    "nest-asyncio>=1.6.0",    # Allow asyncio.run() in Jupyter/running loops
    "python-dotenv",           # Load .env files
    "uvicorn",                 # ASGI server
    "google-generativeai",     # Google Gemini API
    "httpx",                   # Async HTTP client
    # crewai (in sushmita_agent's own pyproject.toml)
]
```

**Why `nest_asyncio`?** When running inside ADK's development server (which already has a running event loop), `asyncio.run()` would normally throw a `RuntimeError`. `nest_asyncio.apply()` patches asyncio to allow nested event loops.

---

## 🆚 ADK vs CrewAI: Side-by-Side

| Feature | Google ADK (Shubham) | CrewAI (Sushmita) |
|---------|---------------------|-------------------|
| Tool Definition | Plain Python function | Class inheriting `BaseTool` |
| Tool Schema | Auto-inferred from type hints | Explicit Pydantic `args_schema` |
| Agent Config | `LlmAgent(instruction=..., tools=[...])` | `Agent(role=..., goal=..., backstory=...)` |
| Execution Model | Event-based streaming `run_async()` | Synchronous `crew.kickoff()` |
| Streaming Support | ✅ Yes | ❌ No (blocking) |
| A2A Compatibility | ✅ Both work equally well via AgentExecutor |  |

---

## 🚀 Running the System

To run this three-agent system locally, you start each agent independently:

```bash
# Terminal 1: Start Shubham's Agent (port 10002)
cd shubham_agent
python -m shubham_agent

# Terminal 2: Start Sushmita's Agent (port 10003)
cd sushmita_agent
python -m sushmita_agent

# Terminal 3: Start Host Agent via ADK Dev UI
cd host_agent
adk web
```

Set your environment variables:
```bash
GOOGLE_API_KEY=your_api_key_here
```

Then navigate to `http://localhost:8000` and ask:
> *"Schedule a badminton game with my friends this week!"*

Watch as the Host Agent:
1. Queries both friend agents simultaneously
2. Cross-references everyone's availability
3. Checks the court schedule
4. Books the court and reports back — all automatically!

---

## 💡 Key Takeaways

### What Makes A2A Powerful

1. **Framework Agnosticism** — Mix ADK, CrewAI, LangGraph, or any custom agent. They all speak A2A.
2. **Loose Coupling** — Agents discover each other at runtime via `AgentCard`. No hardcoded dependencies.
3. **Standardized Communication** — JSON-RPC over HTTP. Any language, any platform can implement an A2A server.
4. **Built-in Task Management** — Task states, artifact storage, and context management are built into the protocol.
5. **Skills and Capabilities** — Agents declare what they can do, enabling intelligent routing and discovery.

### When to Use Multi-Agent Systems

- **Separation of Concerns**: Different agents own different data domains (each friend's calendar is private)
- **Parallel Processing**: Host can query multiple agents simultaneously
- **Independent Scaling**: Each agent service scales independently
- **Team Ownership**: Different teams can build and maintain their own agents
- **Framework Freedom**: Use the best tool for each job

### Limitations to Be Aware Of

- **Latency**: Network calls add overhead. Multi-agent systems are slower than monolithic ones.
- **Complexity**: Debugging distributed systems is harder. You need good logging.
- **State Management**: Keeping context consistent across agents requires careful design.
- **In-Memory Stores**: This demo uses in-memory storage — production systems need persistent databases.

---

## 🔮 What's Next?

This demo is intentionally minimal to illustrate the core patterns. You could extend it to:

- **Add a Kaitlyn Agent** (the `pyproject.toml` even has a commented-out LangGraph dependency!)
- **Persist data** using PostgreSQL or Redis instead of in-memory stores
- **Add authentication** — A2A supports API key and OAuth2 authentication on AgentCards
- **Deploy to cloud** — Each agent is a separate ASGI app, perfect for containerized deployment on Cloud Run
- **Add streaming to Sushmita** — Refactor the CrewAI agent to use async task streaming
- **Multi-modal support** — A2A's `FilePart` allows passing images, documents, and audio between agents

---

## 🎯 Conclusion

The Agent2Agent Protocol solves a fundamental problem in the AI ecosystem: **how do independently-built AI agents communicate and collaborate?**

Through this badminton scheduling demo, we've seen:

- How a **Host Agent** discovers and orchestrates multiple remote agents
- How **AgentCard** enables dynamic agent discovery without hardcoded dependencies
- How the **A2AStarletteApplication** turns any Python agent into an A2A-compatible HTTP server
- How **different frameworks** (ADK and CrewAI) can interoperate seamlessly behind the protocol
- How the **Task lifecycle** (submit → work → complete) provides reliable state management
- How **Artifacts and Parts** form a flexible, multi-modal response format

The future of AI isn't one monolithic agent that does everything — it's a **network of specialized agents**, each an expert in their domain, collaborating via open protocols like A2A.

The badminton game is just the beginning. 🏸

---

*If you found this helpful, give it a clap 👏 and follow for more deep dives into AI agent architecture. Got questions about the code? Drop them in the comments!*

*The full code is available on GitHub: [A2A-Agent2Agent-Protocol-demo](https://github.com/KillerStrike17/A2A-Agent2Agent-Protocol-demo)*

---

**Tags:** `#AI` `#MachineLearning` `#Python` `#GoogleADK` `#AgentToAgent` `#MultiAgent` `#CrewAI` `#A2AProtocol` `#AIEngineering` `#LLM`
