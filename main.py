import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

############################################################
# Callback function called on update config
############################################################


def config(configuration: ConfigClass):
    # Environmental variables can be defined here.
    # example: MY_VAR = configuration.get("MY_VAR", default_value="some default value")
    pass

############################################################
# Callback function called on each execution pass
############################################################


def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    # Loading pre-defined responses
    responses = [
        "The laws of physics describe how matter and energy interact in the universe.",
        "A black hole is a region of spacetime exhibiting gravitational acceleration so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from it.",
        "The theory of evolution by natural selection is a scientific explanation of how species change over time.",
        "The study of the universe is called astronomy.",
        "The scientific study of the structure and behavior of matter and energy is called physics.",
        "The branch of biology that deals with the study of plants is called botany.",
        "The study of how the human body works is called anatomy and physiology.",
        "The study of the behavior and mental processes of animals, including humans, is called psychology.",
        "The study of the physical, chemical, and biological properties of oceans and marine environments is called oceanography.",
        "The study of the Earth's physical characteristics, atmosphere, and oceans, and the processes that shape the planet is called geology.",
    ]

    # Initializing TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(responses)

    output = []
    for text in request.text:
        query_vector = vectorizer.transform([text])
        cosine_similarities = np.dot(query_vector, tfidf.T).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]

        response = responses[related_docs_indices[0]]
        output.append(response)

    return SimpleText(dict(text=output))
