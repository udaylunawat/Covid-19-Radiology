FROM tensorflow/tensorflow:2.3.0

RUN apt-get update -y && apt-get upgrade -y

# setting work directory and copying content
WORKDIR /app
ADD . /app

# Generating data ETL, downloading inference and installing retinanet from source
RUN make -s ETL

EXPOSE 8080

CMD streamlit run serve/app.py --server.port 8080 --server.enableXsrfProtection=false --server.enableCORS=false