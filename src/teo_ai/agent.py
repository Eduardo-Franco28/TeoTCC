import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from teo_ai import integramente, knowledge, llm
from teo_ai.auth import TeoUser
from teo_ai.config import settings

SYSTEM_PROMPT = """
Você é o Teo, o assistente do IntegraMente, um app educacional para crianças e adolescentes autistas.
Você conversa com adultos: pais, responsáveis e profissionais. Responda sempre em português do Brasil,
com frases curtas, tom acolhedor e sem jargão.

O que você faz:
- Tira dúvidas sobre autismo. Quando a base de conhecimento tiver algo sobre a pergunta, os trechos chegam
  junto com ela. Responda só com o que os trechos trouxerem e cite as fontes no fim, numeradas, com título e
  link. Se precisar de mais, use search_knowledge com a dúvida reescrita. Se não houver nada útil, diga que
  não tem essa informação na sua base e sugira conversar com um profissional.
- Cria atividades no IntegraMente de três tipos. Use list_subjects para achar o id da matéria:
  completar a palavra (create_letters_draft, matéria Linguagens); soma com bolinhas (create_sum_draft,
  matéria Matemática, resultado até 30); e agrupar, em que a criança arrasta cada item para o grupo certo
  (create_groups_draft, na matéria que combinar com o tema). Toda atividade nasce como rascunho. Depois de criar,
  diga que ela aparece no cartão de prévia para o adulto revisar e aprovar. Nunca diga que já está no app
  da criança antes da aprovação.
- Quando o adulto quiser atividades prontas ou ideias, use search_catalog para ver o que outros adultos
  publicaram. Diga o título e o autor e explique que ele pode ver a prévia e adicionar pela aba Catálogo,
  aqui no Teo. Atividade adicionada do catálogo não precisa de aprovação: ela vai direto para o app da criança.
- Se a ferramenta devolver erro, ajuste a atividade de acordo com a mensagem e tente de novo.
- Nunca mostre ao adulto nomes técnicos de ferramentas ou campos, como blanks, subjectId ou create_letters_draft.
  Fale "letras que ficam vazias", "matéria" e "atividade".

O que você nunca faz:
- Não faz diagnóstico, não indica remédio nem dose e não substitui terapia ou acompanhamento profissional.
  Quando perguntarem se a criança é autista, acolha a preocupação, explique com carinho que só uma avaliação
  profissional pode responder isso e indique procurar o pediatra, que pode encaminhar para neuropediatra,
  psiquiatra infantil ou psicólogo. Se a base trouxer informação útil sobre sinais de alerta ou direitos,
  use e cite; se não trouxer, não diga que não tem a informação, apenas oriente o próximo passo.
- Não pede nem repete nome completo, escola, endereço ou outros dados que identifiquem a criança.
- Se o adulto falar em crise, desespero ou risco de alguém se machucar, acolha e indique o CVV (ligue 188,
  24 horas) ou o SAMU (192) em emergência.
""".strip()

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "Busca trechos da base de conhecimento do Teo sobre autismo, direitos e educação inclusiva.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A dúvida reescrita como uma pergunta clara"},
            },
            "required": ["query"],
        },
    },
}

DRAFT_TOOLS = {"create_letters_draft", "create_sum_draft", "create_groups_draft"}


@dataclass
class AgentResult:
    reply: str
    drafts: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


def to_openai_tool(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    }


TOOL_STATUS = {
    "search_knowledge": "Pesquisando…",
    "list_subjects": "Consultando as matérias…",
    "search_catalog": "Procurando no catálogo…",
    "list_my_drafts": "Olhando suas atividades…",
    "create_letters_draft": "Criando a atividade…",
    "create_sum_draft": "Criando a atividade…",
    "create_groups_draft": "Criando a atividade…",
}

CONTEXT_PROMPT = "Trechos da base de conhecimento encontrados para a pergunta do adulto. Use só o que for relevante:"


def cite(source: knowledge.Source) -> dict:
    return {"title": source.title, "link": source.link, "page": source.page}


def add_sources(result: AgentResult, sources: list[knowledge.Source]) -> None:
    for source in sources:
        cited = cite(source)
        if cited not in result.sources:
            result.sources.append(cited)


def excerpts(sources: list[knowledge.Source]) -> str:
    return json.dumps([{**cite(source), "content": source.content} for source in sources], ensure_ascii=False)


def mentions(reply: str, sources: list[knowledge.Source]) -> bool:
    text = reply.casefold()
    return "fonte" in text or any(source.title.casefold() in text for source in sources)


async def run_search(query: str, result: AgentResult) -> str:
    sources = await knowledge.search(query)

    if not sources:
        return "Nenhum trecho relevante encontrado na base."

    add_sources(result, sources)
    return excerpts(sources)


async def find_context(message: str) -> list[knowledge.Source]:
    try:
        return await knowledge.search(message)
    except Exception:
        return []


async def chunks(first: Any, iterator: AsyncIterator[Any]) -> AsyncIterator[Any]:
    yield first
    async for chunk in iterator:
        yield chunk


async def stream_step(messages: list[dict], tools: list[dict], state: dict) -> AsyncIterator[dict]:
    first, iterator = await llm.chat_stream(messages, tools=tools, tool_choice="auto", temperature=0.3)
    content: list[str] = []
    calls: dict[int, dict] = {}

    async for chunk in chunks(first, iterator):
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if delta.content:
            content.append(delta.content)
            yield {"type": "delta", "text": delta.content}

        for tool_call in delta.tool_calls or []:
            slot = calls.setdefault(tool_call.index, {"id": "", "name": "", "arguments": ""})
            if tool_call.id:
                slot["id"] = tool_call.id
            if tool_call.function and tool_call.function.name:
                slot["name"] += tool_call.function.name
            if tool_call.function and tool_call.function.arguments:
                slot["arguments"] += tool_call.function.arguments

    state["content"] = "".join(content)
    state["calls"] = [calls[index] for index in sorted(calls)]


async def run_stream(user: TeoUser, history: list[dict], message: str) -> AsyncIterator[dict]:
    result = AgentResult(reply="")
    yield {"type": "status", "text": "Pensando…"}

    context_task = asyncio.create_task(find_context(message))

    async with integramente.open_session(user.token) as session:
        listed = await session.list_tools()
        tools = [SEARCH_TOOL] + [to_openai_tool(tool) for tool in listed.tools]
        context = await context_task

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        if context:
            messages.append({"role": "system", "content": f"{CONTEXT_PROMPT}\n{excerpts(context)}"})
        messages.append({"role": "user", "content": message})

        for step in range(settings.max_agent_steps):
            if step > 0:
                yield {"type": "status", "text": "Pensando…"}

            state: dict = {}
            async for event in stream_step(messages, tools, state):
                yield event

            if not state["calls"]:
                result.reply = state["content"]
                if context and mentions(result.reply, context):
                    add_sources(result, context)
                yield {"type": "done", "result": result}
                return

            yield {"type": "reset"}
            messages.append({
                "role": "assistant",
                "content": state["content"] or None,
                "tool_calls": [
                    {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
                    for call in state["calls"]
                ],
            })

            for call in state["calls"]:
                yield {"type": "status", "text": TOOL_STATUS.get(call["name"], "Trabalhando nisso…")}

                try:
                    arguments = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = None

                if arguments is None:
                    content = "Os argumentos não vieram em JSON válido. Tente de novo."
                elif call["name"] == SEARCH_TOOL["function"]["name"]:
                    content = await run_search(arguments.get("query", message), result)
                else:
                    tool_result = await session.call_tool(call["name"], arguments)
                    content = integramente.result_text(tool_result)

                    if call["name"] in DRAFT_TOOLS and not tool_result.is_error:
                        result.drafts.append(json.loads(content))

                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

    result.reply = "Não consegui concluir agora. Pode reformular o pedido?"
    yield {"type": "done", "result": result}


async def run(user: TeoUser, history: list[dict], message: str) -> AgentResult:
    result = AgentResult(reply="Não consegui concluir agora. Pode reformular o pedido?")

    async for event in run_stream(user, history, message):
        if event["type"] == "done":
            result = event["result"]

    return result
