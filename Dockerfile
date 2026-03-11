FROM public.ecr.aws/lambda/python:3.12

# Poppler — required by pdf2image
RUN dnf install -y poppler-utils && dnf clean all

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["app.handler"]
