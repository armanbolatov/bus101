import pandas as pd
import pickle
import openai
from tqdm import tqdm
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from typing import List, Any
tqdm.pandas()


app = FastAPI()
PRODUCTS_PATH = "data/products.tsv"
IMAGES_PATH = "data/images/"


def check_user_prompt(user_prompt: str) -> str:
    """
    Args:
        user_prompt: the user prompt to check
    Returns:
        answer:      the answer from the GPT-3 model
    """
    prompt = f"Check if {user_prompt} is an appropriate description of a person. Write 'True' if it is, 'False' if it is not. Follow the instructions carefully."
    # get the justification
    answer = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helping a customer find a gift for a friend."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2,
        top_p=1,
    ).choices[0].message.content

    answer = True if answer.startswith("True") else False

    return answer


def embedding_from_string(
    string: str,
    save: bool,
    model: str = "text-embedding-ada-002",
) -> List[float]:
    """
    Args:
        string:    the string to compute the embedding for
        save:      whether to save the embedding to the cache
        model:     the model to use for computing the embedding
    Returns:
        embedding: the embedding of the string
    """
    if save:
        # set the path to the cache pickle file
        embedding_cache_path = "cache/recommendations_cache.pkl"
        # load the cache, or create it if it doesn't exist
        try:
            embedding_cache = pd.read_pickle(embedding_cache_path)
        except FileNotFoundError:
            embedding_cache = {}
            with open(embedding_cache_path, "wb") as embedding_cache_file:
                pickle.dump(embedding_cache, embedding_cache_file)
        # return the embedding from the cache or compute it
        if (string, model) not in embedding_cache.keys():
            embedding = get_embedding(string, model)
            embedding_cache[(string, model)] = embedding
            with open(embedding_cache_path, "wb") as embedding_cache_file:
                pickle.dump(embedding_cache, embedding_cache_file)
        else:
            embedding = embedding_cache[(string, model)]
    else:
        embedding = get_embedding(string, model)
    
    return embedding


def get_justification(name_description: str, user_prompt: str) -> str:
    """
    Args:
        name_description: the name and description of the product
        user_prompt:      the customers' prompt
    Returns:
        justification:    the justification for the product
    """
    prompt = f"Explain why {name_description} is a perfect gift choice for someone who meets the following criteria: \"{user_prompt}\". Keep it brief and concise, and avoid using the words 'excellent'. Also, do not repeat the product name or description. Adhere to this rule diligently!"
    justification = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=100,
        top_p=1,
    ).choices[0].message.content

    return justification


@app.get("/images/{image_id}")
async def get_image(image_id: int):
    """
    Args:
        image_id:    the id of the image
    Returns:
        image:       the image
    """
    image_path = f"{IMAGES_PATH}{image_id}.png"
    return FileResponse(image_path)


@app.post("/recommendations/", response_model=Any)
async def get_recommendations(
    user_prompt: str,
    price_min: int = 0,
    price_max: int = 100500,
    N: int = 5,
):
    """
    Args:
        user_prompt:  the user prompt
        price_min:    the minimum price
        price_max:    the maximum price
        N:            the number of recommendations to return
        products_path: the path to the products file
    Returns:
        recommendations: the recommendations
    """
    if not check_user_prompt(user_prompt):
        return {"error": "The user prompt is not appropriate. Please try again."}
    products = pd.read_csv(PRODUCTS_PATH, sep="\t")
    names_descriptions = products["name"] + ". " + products["description"]
    product_embeddings = names_descriptions.apply(
        lambda x: embedding_from_string(x, save=True)
    )
    prompt_embedding = embedding_from_string(
        "the best gift for " + user_prompt, save=False
    )
    distances = distances_from_embeddings(prompt_embedding, product_embeddings)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    recommendations = products.iloc[indices_of_nearest_neighbors]
    recommendations = recommendations[
        (recommendations["price_min"] >= price_min)
        & (recommendations["price_max"] <= price_max)
    ]
    recommendations = recommendations.iloc[:N]
    recommendations["justification"] = recommendations.progress_apply(
        lambda x: get_justification(x["name"] + ". " + x["description"], user_prompt),
        axis=1,
    )
    recommendations = recommendations.to_dict(orient="records")

    return recommendations