FROM mambaorg/micromamba:2.0.5

USER root
WORKDIR /workspace
COPY environment.service.yml /tmp/environment.service.yml
RUN micromamba install --yes --name base --file /tmp/environment.service.yml \
    && micromamba clean --all --yes

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY datasets ./datasets
RUN micromamba run -n base python -m pip install --no-cache-dir '.[agent,serve]'

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD micromamba run -n base python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"
CMD ["micromamba", "run", "-n", "base", "qryeval", "serve", "--config", "configs/service/local.json", "--host", "0.0.0.0"]
