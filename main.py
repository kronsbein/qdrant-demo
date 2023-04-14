import streamlit as st

from neural_search import NeuralSearch
from conf import COLLECTION_NAME

neural = NeuralSearch(COLLECTION_NAME)


def run_demo():
    st.set_page_config(
        page_title="arxiv publication search",
    )

    st.title("Find similar publications on arxiv based on publication summary")
    st.caption(
        "This demo provides a search interface for arxiv publications based on similar publication summaries."
        " It has been built with the following tools: \n"
        "- Qdrant, a vector similarity engine (https://qdrant.tech/)\n"
        "- Streamlit, a frontend layer for data apps (https://streamlit.io)\n"
        "- arxiv API (python wrapper) (https://pypi.org/project/arxiv/)"
    )

    st.caption("Made by kronsbein (https://kronsbein.github.io)")

    # add text search 
    search = st.text_input(
        "Enter text to find similar publications",
        placeholder="Start searching..."
    )

    st.caption(
        "Check out the most recent publications in AI: https://arxiv.org/list/cs.AI/recent \n"
        " to get an idea on search terms."
    )

    # check search
    if search:
        payload = neural.search(search)
        if payload:
            st.header("Search results:")
            st.divider()
            for idx, item in enumerate(payload):
                # display search results
                st.subheader(str(idx+1) + ". " + item["title"])
                st.caption("Published: " + item["published"][:11])
                st.caption("Check out publication: " + item["entry_id"])   

                # add expander component for summary text
                with st.expander("Expand publication summary"):
                    st.caption(item["summary"]) 

                st.divider() 

            #st.button("Clear")  
        else: 
            st.caption("No results found for search, try different search.") 



if __name__ == "__main__":
    run_demo()
