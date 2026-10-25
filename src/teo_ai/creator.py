import json
import random
import re
import unicodedata

from teo_ai import integramente, llm
from teo_ai.auth import TeoUser

WORD_PATTERN = re.compile(r"^[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]{2,8}$")
VOWELS = set("AEIOUÁÀÂÃÉÊÍÓÔÕÚ")
FALLBACK_EMOJI = "⭐"
LANGUAGES_SUBJECT = "linguagens"
MATH_SUBJECT = "matematica"
MAX_GROUP_NAME = 14
MAX_ITEM_NAME = 18
MAX_QUESTION = 60

WORDS_PROMPT = """
Você ajuda a montar atividades de completar a palavra para crianças e adolescentes autistas.
Tema: {theme}
Escolha {count} palavras em português do Brasil ligadas ao tema. Regras para cada palavra:
- substantivo concreto e comum, que um único emoji represente bem;
- de 3 a 8 letras, só letras (sem espaço, hífen ou número);
- escrita correta, com acentos e cedilha quando a palavra tiver (ex.: CAMARÃO, MAÇÃ, PÉ);
- sem palavras repetidas.
Responda só com JSON, sem texto antes ou depois, neste formato:
{{"title": "título curto da atividade", "words": [{{"word": "GATO", "emoji": "🐱"}}]}}
""".strip()

GROUPS_PROMPT = """
Você ajuda a montar atividades de agrupar para crianças e adolescentes autistas: a criança arrasta cada item para o grupo certo.
Tema: {theme}
Crie {groups} grupos com {items} itens em cada um. Regras:
- cada item pertence a um único grupo, sem dúvida nenhuma (nada que caiba em dois grupos);
- itens concretos e comuns, que um único emoji represente bem;
- nome do grupo com até 14 letras e nome do item com até 18, em português do Brasil, com acentos;
- sem itens repetidos.
Escreva também uma pergunta curta para a criança, com até 40 letras, como "Onde eles vivem?".
Responda só com JSON, sem texto antes ou depois, neste formato:
{{"title": "título curto da atividade", "question": "Onde eles vivem?", "groups": [{{"name": "Mar", "items": [{{"name": "Peixe", "emoji": "🐟"}}]}}]}}
""".strip()

EMOJI_PROMPT = """
Para cada palavra em português abaixo, escolha um único emoji que a represente para uma criança.
Palavras: {words}
Responda só com JSON, sem texto antes ou depois, neste formato:
{{"GATO": "🐱"}}
""".strip()


class CreationError(Exception):
    pass


def normalize_word(word: str) -> str:
    return "".join(word.split()).upper()


def auto_hidden(word: str) -> int:
    if len(word) <= 3:
        return 1
    if len(word) <= 5:
        return 2
    return 3


def pick_blanks(word: str, hidden: int | None) -> list[int]:
    limit = min(len(word) - 1, len(set(word)))
    target = max(1, min(hidden or auto_hidden(word), limit))

    order = sorted(range(len(word)), key=lambda index: (word[index] not in VOWELS, index))
    chosen: list[int] = []
    used: set[str] = set()

    for index in order:
        if word[index] in used:
            continue
        chosen.append(index + 1)
        used.add(word[index])
        if len(chosen) == target:
            break

    return sorted(chosen)


def plain(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(char) != "Mn"
    )


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("resposta sem JSON")
    return json.loads(match.group(0))


async def ask_json(prompt: str) -> dict:
    response = await llm.chat([{"role": "user", "content": prompt}], temperature=0.4)
    return parse_json(response.choices[0].message.content)


async def words_from_theme(theme: str, count: int) -> tuple[str, list[dict]]:
    for _ in range(2):
        try:
            data = await ask_json(WORDS_PROMPT.format(theme=theme, count=count))
        except (ValueError, json.JSONDecodeError):
            continue

        words: list[dict] = []
        seen: set[str] = set()
        for item in data.get("words", []):
            word = normalize_word(str(item.get("word", "")))
            if WORD_PATTERN.match(word) and word not in seen:
                seen.add(word)
                words.append({"word": word, "emoji": str(item.get("emoji") or FALLBACK_EMOJI).strip()})

        if words:
            title = str(data.get("title") or theme).strip()[:60]
            return title, words[:count]

    raise CreationError("O Teo não conseguiu pensar em palavras para esse tema agora. Tente de novo ou digite as palavras.")


async def emojis_for(words: list[str]) -> dict[str, str]:
    try:
        data = await ask_json(EMOJI_PROMPT.format(words=", ".join(words)))
    except Exception:
        return {}

    return {normalize_word(str(key)): str(value).strip() for key, value in data.items() if value}


async def list_subjects(session) -> list[dict]:
    return json.loads(integramente.result_text(await session.call_tool("list_subjects", {})))


async def subject_id_by_name(session, name: str) -> int:
    subjects = await list_subjects(session)

    for subject in subjects:
        if plain(subject["name"]).startswith(name):
            return subject["id"]

    if not subjects:
        raise CreationError("Nenhuma matéria cadastrada no IntegraMente")

    return subjects[0]["id"]


async def create_letters(
    user: TeoUser,
    theme: str | None,
    typed_words: list[str],
    count: int,
    hidden: int | None,
) -> dict:
    if typed_words:
        words = []
        for raw in typed_words:
            word = normalize_word(raw)
            if not WORD_PATTERN.match(word):
                raise CreationError(f"A palavra {raw.strip() or '(vazia)'} precisa ter de 2 a 8 letras, só letras")
            if word not in [item["word"] for item in words]:
                words.append({"word": word, "emoji": ""})

        emojis = await emojis_for([item["word"] for item in words])
        for item in words:
            item["emoji"] = emojis.get(item["word"]) or FALLBACK_EMOJI

        title = (theme or "").strip()[:60] or "Complete a palavra"
    else:
        title, words = await words_from_theme(theme.strip(), count)

    payload = [
        {"word": item["word"], "blanks": pick_blanks(item["word"], hidden), "emoji": item["emoji"]}
        for item in words
    ]

    async with integramente.open_session(user.token) as session:
        subject_id = await subject_id_by_name(session, LANGUAGES_SUBJECT)
        result = await session.call_tool(
            "create_letters_draft",
            {"title": title, "subjectId": subject_id, "words": payload},
        )

    return draft_from(result)


def draft_from(result) -> dict:
    text = integramente.result_text(result)

    if result.is_error:
        raise CreationError(text.splitlines()[0] if text else "Não foi possível criar a atividade")

    return json.loads(text)


def pick_sums(count: int, max_total: int) -> list[dict]:
    min_total = 2 if max_total <= 10 else max_total // 2
    pairs = sorted(
        {(first, second) for first in range(1, max_total) for second in range(first, max_total)
         if min_total <= first + second <= max_total}
    )
    chosen = random.sample(pairs, min(count, len(pairs)))

    sums = []
    for first, second in chosen:
        if random.random() < 0.5:
            first, second = second, first
        sums.append({"first": first, "second": second})

    return sums


async def create_sum(user: TeoUser, count: int, max_total: int) -> dict:
    sums = pick_sums(count, max_total)

    async with integramente.open_session(user.token) as session:
        subject_id = await subject_id_by_name(session, MATH_SUBJECT)
        result = await session.call_tool(
            "create_sum_draft",
            {"title": f"Somas até {max_total}", "subjectId": subject_id, "sums": sums},
        )

    return draft_from(result)


def clean_groups(data: dict, groups: int, items: int) -> tuple[str, str, list[dict]]:
    question = str(data.get("question") or "").strip()
    if not question or len(question) > MAX_QUESTION:
        raise ValueError("pergunta inválida")

    seen_groups: set[str] = set()
    seen_items: set[str] = set()
    result = []

    for group in (data.get("groups") or [])[:groups]:
        name = str(group.get("name") or "").strip()
        if not name or len(name) > MAX_GROUP_NAME or plain(name) in seen_groups:
            raise ValueError("grupo inválido")
        seen_groups.add(plain(name))

        clean_items = []
        for item in group.get("items") or []:
            item_name = str(item.get("name") or "").strip()
            emoji = str(item.get("emoji") or "").strip()
            if not item_name or len(item_name) > MAX_ITEM_NAME or not emoji or plain(item_name) in seen_items:
                continue
            seen_items.add(plain(item_name))
            clean_items.append({"name": item_name, "emoji": emoji})

        if len(clean_items) < items:
            raise ValueError("grupo com poucos itens")

        result.append({"name": name, "items": clean_items[:items]})

    if len(result) != groups:
        raise ValueError("quantidade de grupos errada")

    title = str(data.get("title") or "").strip()[:60]
    return title, question, result


async def groups_from_theme(theme: str, groups: int, items: int) -> tuple[str, str, list[dict]]:
    for _ in range(2):
        try:
            data = await ask_json(GROUPS_PROMPT.format(theme=theme, groups=groups, items=items))
            return clean_groups(data, groups, items)
        except (ValueError, json.JSONDecodeError, AttributeError, TypeError):
            continue

    raise CreationError("O Teo não conseguiu montar os grupos para esse tema agora. Tente de novo ou mude o tema.")


async def create_groups(user: TeoUser, theme: str, subject_id: int, groups: int, items: int) -> dict:
    title, question, payload = await groups_from_theme(theme.strip(), groups, items)

    async with integramente.open_session(user.token) as session:
        result = await session.call_tool(
            "create_groups_draft",
            {"title": title or theme.strip()[:60], "subjectId": subject_id, "question": question, "groups": payload},
        )

    return {"draft": draft_from(result), "question": question, "groups": payload}


async def subjects(user: TeoUser) -> list[dict]:
    async with integramente.open_session(user.token) as session:
        return await list_subjects(session)
