# First stage: Build dependencies in a temporary container
FROM python:3.12.9-slim as builder

WORKDIR /app

# Copy only requirements.txt to leverage Docker cache
COPY requirements.txt .

# Install dependencies in a temporary directory
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Second stage: Final lightweight container
FROM python:3.12.9-slim

WORKDIR /app

# Copy only the installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy the application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
