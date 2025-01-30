# Use Ubuntu as base image
FROM ubuntu:latest

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y \
     curl \
     python3 \
     python3-pip \
     python3-venv

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Set environment variables for the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port that Streamlit will use
EXPOSE 8501

# Copy the model files into the container
COPY /c/Users/lokesh/.ollama/models /root/.ollama/models

# Copy the project files into the container
COPY . .

# Install required Python packages into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
