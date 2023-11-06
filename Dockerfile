# Start with a Python 3.10 base image
FROM python:3.10.6-slim

# Set the working directory in the container
WORKDIR /edge

# Install Poetry
# Poetry provides a custom installer that is recommended for isolated environments.
RUN pip install --no-cache-dir poetry

# Copy the pyproject.toml (and possibly poetry.lock) file to the working directory
COPY pyproject.toml poetry.lock* /edge/

# Install project dependencies
# The `--no-root` flag tells Poetry to install only the dependencies and not the project package itself
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

# Copy the content of the local src directory to the working directory
COPY src/ .

# Command to run on container start
CMD [ "python", "./pipeline.py" ]
