import json
from typing import List, Dict

import chromadb
from chromadb.config import Settings


def compare_semantic_ids(
        client_chroma,
        semantic_id: str
) -> List[Dict[str, str]]:
    """

    """
    collections = client_chroma.list_collections()
    results = []
    for collection in collections:
        collection = client_chroma.get_collection(collection.name)
        # look if semantic id of requested submodel element is in database
        try:
            result = collection.get(where={"SESemanticId": semantic_id})
            value = result['documents'][0]
            value_dict = json.loads(value)
            final_result = {
                "matching_method": "Semantic equivalent, same semantic Id",
                "matching_algorithm": "None",
                "matching_distance": 0,
                "aas_id": result['metadatas'][0]['AASId'],
                "aas_id_short": result['metadatas'][0]['AASIdShort'],
                "submodel_id_short": result['metadatas'][0]['SubmodelName'],
                "submodel_id": result['metadatas'][0]['SubmodelId'],
                "id_short_path": result['metadatas'][0]['IdShortPath'],
                "matched_object": value_dict
            }
            # print(final_result)

        except Exception:
            final_result = 0
        results.append(final_result)

    return results


def compare_semantic_ids_one_aas(collection, client_chroma, preferred_name, definition, unit, datatype, semantic_id):
    # look if semantic id of requested submodel element is in database
    try:
        result = collection.get(where={"SESemanticId": semantic_id})
        value = result['documents'][0]
        value_dict = json.loads(value)
        final_result = {
            "matching_method": "Semantic equivalent, same semantic Id",
            "matching_algorithm": "None",
            "matching_distance": 0,
            "aas_id": result['metadatas'][0]['AASId'],
            "aas_id_short": result['metadatas'][0]['AASIdShort'],
            "submodel_id_short": result['metadatas'][0]['SubmodelName'],
            "submodel_id": result['metadatas'][0]['SubmodelId'],
            "id_short_path": result['metadatas'][0]['IdShortPath'],
            "matched_object": value_dict
        }
        # print(final_result)

    except Exception:
        final_result = 0

    return final_result
