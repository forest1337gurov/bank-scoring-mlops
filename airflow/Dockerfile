FROM apache/airflow:2.10.3

USER root
# Если apt-пакеты не нужны — этот блок можно убрать полностью
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#        build-essential \
#     && apt-get autoremove -yqq --purge \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt