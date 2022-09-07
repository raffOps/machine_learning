docker build -t jupyter .
docker run -v ${PWD}:/home/jovyan -p 8888:8888 jupyter