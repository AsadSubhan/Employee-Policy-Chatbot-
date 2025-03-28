# Use Python  as the base image
FROM python:3.12.9

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

