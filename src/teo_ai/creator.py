import json
import re
import unicodedata

from teo_ai import integramente, llm
from teo_ai.auth import TeoUser
from teo_ai.config import settings

WORD_PATTERN = re.compile(r"^[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]{2,8}$")
VOWELS = set("AEIOUÁÀÂÃÉÊÍÓÔÕÚ")
FALLBACK_EMOJI = "⭐"
LANGUAGES_SUBJECT = "linguagens"

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
    response = await llm.client().chat.completions.create(
        model=settings.teo_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
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


async def languages_subject_id(session) -> int:
    subjects = json.loads(integramente.result_text(await session.call_tool("list_subjects", {})))

    for subject in subjects:
        if plain(subject["name"]).startswith(LANGUAGES_SUBJECT):
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
        subject_id = await languages_subject_id(session)
        result = await session.call_tool(
            "create_letters_draft",
            {"title": title, "subjectId": subject_id, "words": payload},
        )

    text = integramente.result_text(result)

    if result.is_error:
        raise CreationError(text.splitlines()[0] if text else "Não foi possível criar a atividade")

    return json.loads(text)
