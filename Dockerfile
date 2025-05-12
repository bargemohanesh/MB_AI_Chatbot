# 1. Base Python image
FROM python:3.10-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy all project files into the container
COPY . .

# 4. Install system dependencies
RUN apt-get update && apt-get install -y unzip awscli && apt-get clean

# 5. Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 6. Set environment variables (OPTIONAL: change region as per your S3 bucket)
ENV AWS_DEFAULT_REGION=ap-south-1

# 7. Expose the port your app will run on (Change if you're using Flask on 5000)
EXPOSE 8000

# 8. Command to run your app
CMD ["python", "main.py"]
