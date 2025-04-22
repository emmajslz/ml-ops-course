# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir mlflow

# Expose the default MLflow port
EXPOSE 5000

# Set the entrypoint to MLflow
ENTRYPOINT ["mlflow"]

# Default command to run MLflow server
CMD ["server", "--host", "0.0.0.0", "--port", "5000"]