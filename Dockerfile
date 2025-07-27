# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY ./flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by the app
RUN python -m nltk.downloader stopwords wordnet

# Copy the rest of the application source code and models
# This places app.py and vectorizer.pkl in the correct locations
COPY ./flask_app/ ./
COPY ./models/ ./models/

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application

CMD ["python", "app.py"]

#Prod Use gunicorn for production, but Flask dev server is fine for this
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]

