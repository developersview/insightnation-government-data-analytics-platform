FROM astrocrpublic.azurecr.io/runtime:3.0-4

# Install custom Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy DAGs and include folders into the container
COPY dags/ /usr/local/airflow/dags/
COPY include/ /usr/local/airflow/include/