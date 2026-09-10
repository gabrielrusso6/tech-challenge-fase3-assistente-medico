FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps .
COPY data ./data
RUN useradd --create-home appuser && mkdir runtime && chown appuser runtime
USER appuser
ENTRYPOINT ["tc3"]
CMD ["--mode", "fixture"]
