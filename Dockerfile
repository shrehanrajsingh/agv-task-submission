# Use an official Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . .

# Install dependencies
RUN pip install ai2thor numpy opencv-python matplotlib
