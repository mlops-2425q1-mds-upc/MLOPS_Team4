# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-venv python3-pip git sudo && \
    rm -rf /var/lib/apt/lists/*

# Copy the contents of your repo into the container
COPY . .

# Set up the virtual environment
RUN python3 -m venv venv

RUN pip install dvc==3.55.2

# Initialize git and DVC, if necessary
ARG DVC_USER
ARG DVC_TOKEN

RUN git init && \
    git remote get-url origin || git remote add origin https://github.com/mlops-2425q1-mds-upc/MLOPS_Team4.git && \
    dvc remote modify origin --local auth basic && \
    dvc remote modify origin --local user "$DVC_USER" && \
    dvc remote modify origin --local password "$DVC_TOKEN" && \
    dvc pull

# Activate the virtual environment and install dependencies
RUN . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Ensure the required directories have the right permissions
RUN sudo chown -R root:root /app/.dvc /app/data /app/models

# Expose the port that uvicorn will use
EXPOSE 5000

# Set the command to start the API with uvicorn
CMD ["sh", "-c", "source venv/bin/activate && PYTHONPATH=./Sentiment_Analysis uvicorn api:app --host 0.0.0.0 --port 5000"]
