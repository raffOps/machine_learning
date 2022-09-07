FROM jupyter/minimal-notebook:latest AS jupyter
CMD ["start-notebook.sh", "--NotebookApp.token=''"]
COPY requirements.txt .
RUN pip install --quiet --no-cache-dir -r requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
