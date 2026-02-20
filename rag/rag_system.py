import os
import pickle
from typing import List, Dict, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pandas as pd
import requests
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

# Chargement des variables d'environnement
load_dotenv()


class MistralConfig:
    MODELS = {
        "devstral-small": {
            "name": "devstral-small-latest",
            "temperature": 0,
            "max_tokens": 1000,
        }
    }

    @classmethod
    def get_model_config(cls, model_key):
        if model_key not in cls.MODELS:
            raise ValueError(f"Modèle inconnu. Choisissez parmi : {list(cls.MODELS.keys())}")
        return cls.MODELS[model_key]


class RAGSystem:
    # recherche et génération de réponses sur les événements culturels de Lille.
    def __init__(
        self,
        index_path = "faiss_index.bin",
        metadata_path = "faiss_metadata.pkl",
        embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    ):

        # Initialise le système RAG.
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name

        # Composants à charger
        self.index: Optional[faiss.Index] = None
        self.chunks: Optional[List[str]] = None
        self.metadata_list: Optional[List[Dict]] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.llm: Optional[ChatMistralAI] = None

        # Chargement au démarrage
        self._load_components()

    def _load_components(self):

        # 1. Charger l'index FAISS
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index FAISS introuvable : {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        # 2. Charger les métadonnées
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Métadonnées introuvables : {self.metadata_path}")
        with open(self.metadata_path, 'rb') as f:
            metadata_store = pickle.load(f)
        self.chunks = metadata_store['chunks']
        self.metadata_list = metadata_store['metadata']

        # 3. Charger le modèle d'embeddings
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.llm = ChatMistralAI(
            model="devstral-small-latest",
            temperature=0,
            max_tokens=1000
        )


    def _create_prompt(self, context, question) :
        template = """Tu es un assistant expert en événements culturels de Lille.
            Tu réponds UNIQUEMENT avec les informations extraites des documents fournis.

            RÈGLES ABSOLUES :
            1. Ta réponse doit toujours être directement liée au sujet précis de la question
            2. Interdiction absolue de commencer par : "Oui", "Non", "D'après", "Selon"
            3. Chaque affirmation doit être tirée d'un document source
            4. N'ajoute AUCUNE information extérieure aux documents
            5. Si l'information n'est pas disponible, propose 3 alternatives similaires
            6. Maximum 5 événements
            7. Format : **[Titre]** - [Description] - Lieu : [lieu]

            DOCUMENTS SOURCES :
            {context}

            QUESTION : {question}"""
    
        return ChatPromptTemplate.from_template(template)

    def _format_context(self, chunks_list, metadata_list):

        formatted_context = ""

        for i, (chunk, meta) in enumerate(zip(chunks_list, metadata_list), 1):
            formatted_context += f"\n[DOCUMENT {i}]\n"
            formatted_context += f"{chunk}\n"
            formatted_context += f"---\n"

        return formatted_context

    # Stopwords français à ignorer dans le keyword scoring
    _STOPWORDS = {
        'les', 'des', 'une', 'pour', 'dans', 'sur', 'avec', 'qui', 'que',
        'est', 'son', 'ses', 'par', 'aux', 'tout', 'plus', 'mais', 'où',
        'quels', 'quelles', 'quel', 'quelle', 'sont', 'ont', 'mes', 'ces',
        'cette', 'votre', 'avoir', 'être', 'peut', 'elle', 'ils', 'elles',
        'puis', 'trouver', 'comment', 'quoi', 'faire', 'lieu', 'lille',
        'mois', 'ceci', 'celui', 'celui-ci', 'aussi', 'comme', 'entre',
    }

    def _keyword_score(self, query, chunk):
        query_words = set(query.lower().split())
        chunk_lower = chunk.lower()
        # Filtre stopwords et mots courts
        query_words = {w.strip('?!.,éèêëàâùûîï') for w in query_words
                       if len(w) > 3 and w.strip('?!.,') not in self._STOPWORDS}
        if not query_words:
            return 0.0
        # Score : proportion de mots-clés présents dans le chunk (substring match)
        matches = sum(1 for w in query_words if w in chunk_lower)
        # Bonus si le chunk contient plusieurs mots-clés (pertinence cumulée)
        bonus = min(matches / max(len(query_words), 1), 1.0)
        return bonus

    def _rerank_chunks(self, query, query_embedding, chunks, metadata, top_k):
        chunk_embeddings = self.embedding_model.encode(chunks)
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        scores = []
        for emb, chunk in zip(chunk_embeddings, chunks):
            e = emb / (np.linalg.norm(emb) + 1e-9)
            cosine = float(np.dot(q, e))
            keyword = self._keyword_score(query, chunk)
            # Score hybride : 60% cosinus + 40% keyword
            hybrid = 0.60 * cosine + 0.40 * keyword
            scores.append(hybrid)
        ranked = sorted(zip(scores, chunks, metadata), key=lambda x: x[0], reverse=True)
        ranked = ranked[:top_k]
        return [c for _, c, _ in ranked], [m for _, _, m in ranked]

    def query(
        self,
        question,
        top_k = 3,
        model_key = "devstral-small"
    ) :
        # 1. Récupération de la config du modèle
        config = MistralConfig.get_model_config(model_key)

        # 2. Encoder la question
        query_embedding = self.embedding_model.encode([question])[0]
        query_vector = query_embedding.astype('float32').reshape(1, -1)

        # 3. Recherche dans l'index FAISS (on récupère plus de candidats pour le reranking)
        candidate_k = min(top_k * 2, self.index.ntotal)
        distances, indices = self.index.search(query_vector, candidate_k)

        # 4. Extraction des chunks et métadonnées candidats
        candidate_chunks = [self.chunks[idx] for idx in indices[0]]
        candidate_metadata = [self.metadata_list[idx] for idx in indices[0]]

        # 5. Reranking hybride (cosinus + keyword) et sélection des top_k
        retrieved_chunks, retrieved_metadata = self._rerank_chunks(
            question, query_embedding, candidate_chunks, candidate_metadata, top_k
        )

        # 6. Formatage du contexte
        context = self._format_context(retrieved_chunks, retrieved_metadata)

        # 7. Création de la chaîne RAG avec LangChain
        try:
            prompt = self._create_prompt(context, question)

            chain = (
                {"context": lambda x: context, "question": lambda x: question}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            response_text = chain.invoke({})

        except Exception as e:
            response_text = f"Erreur lors de l'appel au modèle {model_key}: {str(e)}"


        # 8. Retour structuré
        return {
            'model': model_key,
            'model_name': config['name'],
            'question': question,
            'answer': response_text,
            'sources': retrieved_metadata,
            'contexts': retrieved_chunks,
            'nb_sources': len(retrieved_chunks),
            'config': config
        }

    def rebuild(self) :
        try:
            # 1. Récupération des événements
            OPENDATASOFT_BASE_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
            DATASET_ID = "evenements-publics-openagenda"
            CITY = "Lille"

            events = []
            rows = 100
            start = 0
            max_results = 500

            while len(events) < max_results:
                params = {
                    'dataset': DATASET_ID,
                    'q': f'location_city:"{CITY}"',
                    'rows': rows,
                    'start': start,
                    'facet': ['location_city', 'location_department']
                }

                response = requests.get(OPENDATASOFT_BASE_URL, params=params)
                response.raise_for_status()

                data = response.json()

                if 'records' not in data or len(data['records']) == 0:
                    break

                for record in data['records']:
                    events.append(record['fields'])

                total = data.get('nhits', 0)
                print(f"  Récupéré {len(events)}/{total} événements...")

                if len(events) >= total or len(data['records']) < rows:
                    break

                start += rows


            # 2. Création du corpus
            df_events = pd.DataFrame(events)
            corpus_df = df_events[['title_fr', 'description_fr', 'location_name', 'location_city', 'location_address']].copy()

            # 3. Génération des chunks
            chunks = []
            metadata_list = []

            for idx, row in corpus_df.iterrows():
                chunk_text = f"""Titre: {row['title_fr']}
                                Description: {row['description_fr']}
                                Lieu: {row['location_name']}, {row['location_city']}"""

                chunks.append(chunk_text)

                metadata_list.append({
                    'event_id': idx,
                    'title': row['title_fr'],
                    'location': row['location_city']
                })

            # 4. Génération des embeddings
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)

            # 5. Création de l'index FAISS
            embeddings_array = embeddings.astype('float32')
            dimension = embeddings_array.shape[1]

            new_index = faiss.IndexFlatL2(dimension)
            new_index.add(embeddings_array)

            # 6. Sauvegarde
            faiss.write_index(new_index, self.index_path)

            metadata_store = {
                'chunks': chunks,
                'metadata': metadata_list
            }

            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata_store, f)

            # 7. Rechargement des composants
            self._load_components()

            return {
                'status': 'success',
                'message': 'Base vectorielle reconstruite avec succès',
                'nb_events': len(events),
                'nb_chunks': len(chunks),
                'dimension': dimension
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Erreur lors de la reconstruction : {str(e)}',
                'nb_events': 0,
                'nb_chunks': 0,
                'dimension': 0
            }
