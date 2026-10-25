# TeoTCC

Backend do Teo, o assistente com IA do IntegraMente. Conversa com adultos sobre autismo usando uma base de conhecimento (RAG) e cria rascunhos de atividades no IntegraMente pelo servidor MCP do backend Spring.

## Stack

- Python 3.12, FastAPI, uv
- Postgres com pgvector (Docker no local, Supabase em produção)
- LLM e embeddings por uma API no padrão OpenAI (NVIDIA)
- Cliente MCP falando com o `app-mobile-tcc`

## Rodando localmente

1. Copie `.env.example` para `.env` e preencha `TCC_JWT_SECRET` (o mesmo do Spring), `LLM_API_KEY`, `TEO_MODEL` e `TEO_EMBEDDING_MODEL`.
2. Suba o banco do Teo (porta 5434):

   ```bash
   docker compose up -d
   ```

3. Instale as dependências e suba a API (porta 8000):

   ```bash
   uv sync
   uv run teo-ai
   ```

O Spring precisa estar rodando em `INTEGRAMENTE_API_URL` para o Teo criar atividades.

## Base de conhecimento

Coloque os documentos (`.pdf`, `.md` ou `.txt`) em `knowledge/` e descreva cada um em `knowledge/fontes.json`:

```json
[
  {
    "arquivo": "lei-12764-2012.pdf",
    "titulo": "Lei 12.764/2012 (Lei Berenice Piana)",
    "link": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2012/lei/l12764.htm",
    "status": "ativo"
  }
]
```

Depois rode:

```bash
uv run teo-ingest
```

Documentos que não mudaram são pulados, e os que saem do `fontes.json` deixam de aparecer na busca.

## Rotas

| Rota | O que faz |
| --- | --- |
| `GET /health` | verifica se a API está no ar |
| `POST /chat` | conversa com o Teo (`message` e, opcionalmente, `conversationId`) |
| `POST /chat/stream` | mesma conversa, mas a resposta chega aos poucos (eventos `status`, `delta`, `reset`, `done`, `error`) |
| `GET /conversations/{id}/messages` | histórico de uma conversa |
| `GET /conversations` | conversas da pessoa, da mais recente para a mais antiga |
| `POST /activities/letters` | cria um rascunho de completar a palavra (`theme` ou `words`, `count`, `hidden`) |
| `POST /activities/sum` | cria um rascunho de soma com bolinhas (`count`, `maxTotal` 10, 20 ou 30) |
| `POST /activities/groups` | cria um rascunho de agrupar (`theme`, `subjectId`, `groups`, `items`) |
| `GET /subjects` | matérias do IntegraMente |

Todas, menos `/health`, exigem o token do Teo emitido pelo `POST /auth/teo/exchange` do Spring.
