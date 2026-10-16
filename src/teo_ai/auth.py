from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException, status

from teo_ai.config import settings

TEO_SCOPE = "teo"
ISSUER = "auth-integramente"


@dataclass(frozen=True)
class TeoUser:
    email: str
    token: str


def current_user(authorization: str | None = Header(default=None)) -> TeoUser:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sessão do Teo não encontrada")

    token = authorization.removeprefix("Bearer ").strip()

    try:
        payload = jwt.decode(
            token,
            settings.tcc_jwt_secret,
            algorithms=["HS256"],
            issuer=ISSUER,
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sessão do Teo inválida ou expirada")

    if payload.get("scope") != TEO_SCOPE:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Esse acesso não libera o Teo")

    return TeoUser(email=payload["sub"], token=token)
