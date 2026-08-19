"""Local production-like serving package."""


def create_app(*args, **kwargs):
    from qryeval_plus.service.app import create_app as factory

    return factory(*args, **kwargs)


__all__ = ["create_app"]
