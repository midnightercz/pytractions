FROM registry.fedoraproject.org/fedora-minimal:latest
ARG SDIST=dist/pytractions-0.0.1.tar.gz
COPY ${SDIST} /
RUN microdnf install -y python3 python3-pip
RUN pip install ${SDIST}
ENTRYPOINT ["python", "-m", "pytractions.cli"]
