import secrets


def new_id(prefix: str) -> str:
    """'a' -> 'a_1f2e3d4c'"""
    return f"{prefix}_{secrets.token_hex(4)}"
