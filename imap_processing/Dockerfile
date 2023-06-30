FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y python3-dev build-essential

RUN python -m pip install --upgrade pip && \
    pip install awscli

# Install poetry
RUN pip install 'poetry==1.4.0'

WORKDIR /imap_processing
# Copy source directory is repo's home path
COPY . .
RUN file="$(ls -1 /imap_processing)" && echo $file

# Install imap-processing using below command
# RUN pip install .

