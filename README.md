# qdrant-demo


This repository contains code for a demo utilising:

- [Qdrant](https://qdrant.tech/) 
- [Streamlit](https://streamlit.io)
- [arxiv API (python wrapper)](https://pypi.org/project/arxiv/)

It provides a text similiarity search based on the *1000* most recent publications from [arxiv](https://arxiv.org). It uses Qdrant's vector database and streamlit as a frontend layer. 

## Requirements


Install python requirements:

```
pip install -r requirements.txt
```

You also need [Docker](https://docs.docker.com/get-docker/) to run Qdrant. 


## How to run


Running this app locally requires to prepare the data first. 

Pull recent Qdrant image:

```
docker pull qdrant/qdrant
```

Now run the service inside Docker:

```
docker run -d -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

After starting the service, upload the data by running: 

```
# Init neural search
python -m init_neural_search
```

Finally, you can launch the application:

```
streamlit run main.py
```

This should expose the application at [http://localhost:8501](http://localhost:8501)