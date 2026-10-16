import json
from dataclasses import dataclass, field

from teo_ai import integramente, knowledge, llm
from teo_ai.auth import TeoUser
from teo_ai.config import settings

SYSTEM_PROMPT = """
Você é o Teo, o assistente do IntegraMente, um app educacional para crianças e adolescentes autistas.
Você conversa com adultos: pais, responsáveis e profissionais. Responda sempre em português do Brasil,
com frases curtas, tom acolhedor e sem jargão.

O que você faz:
- Tira dúvidas sobre autismo usando a ferramenta search_knowledge. Responda só com o que ela trouxer e cite
  as fontes no fim, numeradas, com título e link. Se ela não trouxer nada útil, diga que não tem essa
  informação na sua base e sugira conversar com um profissional.
- Cria atividades no IntegraMente. Hoje existe só o tipo "completar a palavra": use list_subjects para achar
  a matéria Linguagens e depois create_letters_draft. Toda atividade nasce como rascunho. Depois de criar,
  diga que ela aparece no cartão de prévia para o adulto revisar e aprovar. Nunca diga que já está no app
  da criança antes da aprovação.
- Se a ferramenta devolver erro, ajuste a atividade de acordo com a mensagem e tente de novo.

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

DRAFT_TOOLS = {"create_letters_draft"}


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


async def run_search(query: str, result: AgentResult) -> str:
    sources = await knowledge.search(query)

    if not sources:
        return "Nenhum trecho relevante encontrado na base."

    payload = []
    for source in sources:
        cited = {"title": source.title, "link": source.link, "page": source.page}
        if cited not in result.sources:
            result.sources.append(cited)
        payload.append({**cited, "content": source.content})

    return json.dumps(payload, ensure_ascii=False)


async def run(user: TeoUser, history: list[dict], message: str) -> AgentResult:
    result = AgentResult(reply="")

    async with integramente.open_session(user.token) as session:
        listed = await session.list_tools()
        tools = [SEARCH_TOOL] + [to_openai_tool(tool) for tool in listed.tools]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": message}]

        for _ in range(settings.max_agent_steps):
            response = await llm.client().chat.completions.create(
                model=settings.teo_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
            )

            answer = response.choices[0].message

            if not answer.tool_calls:
                result.reply = answer.content or ""
                return result

            messages.append(answer.model_dump(exclude_none=True))

            for call in answer.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = None

                if arguments is None:
                    content = "Os argumentos não vieram em JSON válido. Tente de novo."
                elif call.function.name == SEARCH_TOOL["function"]["name"]:
                    content = await run_search(arguments.get("query", message), result)
                else:
                    tool_result = await session.call_tool(call.function.name, arguments)
                    content = integramente.result_text(tool_result)

                    if call.function.name in DRAFT_TOOLS and not tool_result.is_error:
                        result.drafts.append(json.loads(content))

                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    result.reply = "Não consegui concluir agora. Pode reformular o pedido?"
    return result
