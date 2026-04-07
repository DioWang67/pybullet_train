# 延遲導入以避免 PyBullet 硬性依賴

def __getattr__(name):
    """延遲導入環境"""
    if name == "CassieEnv":
        from envs.cassie_env import CassieEnv
        return CassieEnv
    elif name == "H1Env":
        from envs.h1_env import H1Env
        return H1Env
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = ["CassieEnv", "H1Env"]
