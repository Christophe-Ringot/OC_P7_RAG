import os
import pickle
from typing import List, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from dotenv import load_dotenv
import pandas as pd
import requests

# Chargement des variables d'environnement
load_dotenv()


class MistralConfig:
    MODELS = {
        "devstral-small": {
            "name": "devstral-small-2505",
            "temperature": 0.3,
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
        self.mistral_client: Optional[Mistral] = None

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

        # 4. Initialiser le client Mistral
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY non définie dans .env")
        self.mistral_client = Mistral(api_key=api_key)


    def _create_mistral_prompt(self, context, question) :
        prompt = f"""<s>[INST] Tu es un assistant expert en événements culturels de Lille.

                    INSTRUCTIONS :
                    - Réponds UNIQUEMENT avec les informations des documents fournis
                    - Si l'information n'existe pas dans les documents, réponds "Information non disponible dans ma base"
                    - Cite TOUJOURS tes sources (titre de l'événement + lieu)
                    - Reste factuel, ne déduis rien, ne brode pas
                    - Réponds en français

                    DOCUMENTS SOURCES :
                    {context}

                    QUESTION :
                    {question}
                    [/INST]</s>

                    RÉPONSE :"""

        return prompt

    def _format_context(self, chunks_list, metadata_list):

        formatted_context = ""

        for i, (chunk, meta) in enumerate(zip(chunks_list, metadata_list), 1):
            formatted_context += f"\n[DOCUMENT {i}]\n"
            formatted_context += f"{chunk}\n"
            formatted_context += f"---\n"

        return formatted_context

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

        # 3. Recherche dans l'index FAISS
        distances, indices = self.index.search(query_vector, top_k)

        # 4. Extraction des chunks et métadonnées
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        retrieved_metadata = [self.metadata_list[idx] for idx in indices[0]]

        # 5. Formatage du contexte
        context = self._format_context(retrieved_chunks, retrieved_metadata)

        # 6. Création du prompt
        prompt = self._create_mistral_prompt(context, question)

        # 7. Appel API Mistral
        try:
            chat_response = self.mistral_client.chat.complete(
                model=config["name"],
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )

            response_text = chat_response.choices[0].message.content

        except Exception as e:
            response_text = f"Erreur lors de l'appel au modèle {model_key}: {str(e)}"

        # 8. Retour structuré
        return {
            'model': model_key,
            'model_name': config['name'],
            'question': question,
            'answer': response_text,
            'sources': retrieved_metadata,
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
