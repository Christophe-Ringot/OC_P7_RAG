import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from rag.rag_system import RAGSystem
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def load_evaluation_dataset(dataset_path):

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Le dataset doit être une liste de dictionnaires")

    return data


def generate_rag_responses(rag_system, questions, top_k=7):

    responses = []

    for question in questions:
        print(f"  Génération de la réponse pour: {question[:50]}...")
        result = rag_system.query(question, top_k=top_k)
        responses.append(result)

    return responses


def prepare_ragas_dataset(eval_data, rag_responses):
    data = {
        'user_input': [],
        'response': [],
        'retrieved_contexts': [],
        'reference': []
    }

    for eval_item, rag_response in zip(eval_data, rag_responses):
        data['user_input'].append(eval_item['question'])
        data['response'].append(rag_response['answer'])

        contexts = rag_response.get('contexts', [])
        data['retrieved_contexts'].append(contexts)

        ground_truth = eval_item.get('ground_truth', '')
        data['reference'].append(ground_truth)

    return Dataset.from_dict(data)


def run_evaluation(dataset):

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    run_config = RunConfig(max_workers=1, timeout=120)

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config
    )

    df = results.to_pandas()
    metric_cols = [col for col in df.columns
                   if col not in ('user_input', 'response', 'retrieved_contexts', 'reference')]
    scores = {col: float(df[col].mean()) for col in metric_cols}

    return scores


def main():
    dataset_path = "./eval_dataset.json"
    print(f"Utilisation du dataset: {dataset_path}")
    eval_data = load_evaluation_dataset(dataset_path)

    rag_system = RAGSystem(
        index_path="./faiss_index.bin",
        metadata_path="./faiss_metadata.pkl"
    )

    questions = [item['question'] for item in eval_data]
    rag_responses = generate_rag_responses(rag_system, questions, top_k=7)

    ragas_dataset = prepare_ragas_dataset(eval_data, rag_responses)

    print(f"Dataset pret : {len(ragas_dataset)} exemples")

    scores = run_evaluation(ragas_dataset)

    print("\n=== Résultats RAGAS ===")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()
