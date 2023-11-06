# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install flask scikit-learn

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]