FROM python:3.8-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools



COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt



COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user transforms.py /opt/app/
COPY --chown=user:user config /opt/app/config
COPY --chown=user:user ckpt_lesions /opt/app/ckpt_lesions

ENTRYPOINT [ "python", "-m", "process" ]
