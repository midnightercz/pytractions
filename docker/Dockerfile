FROM registry.fedoraproject.org/fedora-minimal:latest
LABEL quay.expires-after=2d
ARG SDIST=dist/pytractions-0.0.1.tar.gz
RUN mkdir dist
COPY ${SDIST} /dist/
RUN microdnf install -y python3 python3-pip
RUN pip install -v --prefix /usr/ ${SDIST}
ENTRYPOINT ["python", "-m", "pytractions.cli"]
