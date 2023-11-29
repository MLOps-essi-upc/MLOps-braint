FROM python:3.8

WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 and 8501 available to the world outside this container
EXPOSE 8000 8501

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["sh", "-c", "streamlit run src/front/ui.py"]
