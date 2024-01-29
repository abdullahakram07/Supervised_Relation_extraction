# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

## Download the file using gdown
#RUN gdown --id 1nL7EEHLbiuFHt7NX_Q-Zf1gS4jgc6cuq --output /content/checkpoint.ckpt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run Uvicorn to start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
