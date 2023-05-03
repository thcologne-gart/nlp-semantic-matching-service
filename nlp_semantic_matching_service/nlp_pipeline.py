import numpy as np
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle

from semantic_matching_interface import interface, query, response

def query_aas(model, metalabel, preferred_name, definition, unit, datatype, collection, number_elements):
        
        datatype_mapping = {
            "boolean": "BOOLEAN",
            "string": "STRING",
            "string_translatable": "STRING",
            "translatable_string": "STRING",
            "non_translatable_string": "STRING",
            "date": "DATE",
            "data_time": "DATE",
            "uri": "URI",
            "int": "INT",
            "int_measure": "INT",
            "int_currency": "INT",
            "integer": "INT",
            "real": "REAL",
            "real_measure": "REAL",
            "real_currency": "REAL",
            "enum_code": "ENUM_CODE",
            "enum_int": "ENUM_CODE",
            "ENUM_REAL": "ENUM_CODE",
            "ENUM_RATIONAL": "ENUM_CODE",
            "ENUM_BOOLEAN": "ENUM_CODE",
            "ENUM_STRING": "ENUM_CODE",
            "enum_reference": "ENUM_CODE",
            "enum_instance": "ENUM_CODE",
            "set(b1,b2)": "SET",
            "constrained_set(b1,b2,cmn,cmx)": "SET",
            "set [0,?]": "SET",
            "set [1,?]": "SET",
            "set [1, ?]": "SET",
            "nan": "NaN",
            "media_type": "LARGE_OBJECT_TYPE",
        }

        unit_mapping = {
            "nan": "NaN",
            "hertz": "FREQUENCY",
            "hz": "FREQUENCY",
            "pa": "PRESSURE",
            "pascal": "PRESSURE",
            "n/m²": "PRESSURE",
            "bar": "PRESSURE",
            "%": "SCALARS_PERC",
            "w": "POWER",
            "watt": "POWER",
            "kw": "POWER",
            "kg/m³": "CHEMISTRY",
            "m²/s": "CHEMISTRY",
            "pa*s": "CHEMISTRY",
            "v": "ELECTRICAL",
            "volt": "ELECTRICAL",
            "db": "ACOUSTICS",
            "db(a)": "ACOUSTICS",
            "k": "TEMPERATURE",
            "°c": "TEMPERATURE",
            "n": "MECHANICS",
            "newton": "MECHANICS",
            "kg/s": "FLOW",
            "kg/h": "FLOW",
            "m³/s": "FLOW",
            "m³/h": "FLOW",
            "l/s": "FLOW",
            "l/h": "FLOW",
            "µm": "LENGTH",
            "mm": "LENGTH",
            "cm": "LENGTH",
            "dm": "LENGTH",
            "m": "LENGTH",
            "meter": "LENGTH",
            "m/s": "SPEED",
            "km/h": "SPEED",
            "s^(-1)": "FREQUENCY",
            "1/s": "FREQUENCY",
            "s": "TIME",
            "h": "TIME",
            "min": "TIME",
            "d": "TIME",
            "hours": "TIME",
            "a": "ELECTRICAL",
            "m³": "VOLUME",
            "m²": "AREA",
            "rpm": "FLOW",
            "nm": "MECHANICS",
            "m/m": "MECHANICS",
            "m³/m²s": "MECHANICS",
            "w(m²*K)": "HEAT_TRANSFER",
            "kwh": "ELECTRICAL",
            "kg/(s*m²)": "FLOW",
            "kg": "MASS",
            "w/(m*k)": "HEAT_TRANSFER",
            "m²*k/w": "HEAT_TRANSFER",
            "j/s": "POWER",
        }

        unit_lower = unit.lower()
        datatype_lower = datatype.lower()

        unit_categ = unit_mapping.get(unit_lower)
        datatype_categ = datatype_mapping.get(datatype_lower)

        if unit_categ == None:
            unit_categ = "NaN"
        if datatype_categ == None:
            datatype_categ = "NaN"
        
        concat = (unit_categ, datatype_categ)
        keys = [k for k, v in metalabel.items() if v == concat]
        metadata = keys[0]
        #print(metadata)

        # Encode Name from submodel element
        name_embedding = model.encode(preferred_name)
        # Encode definitiob
        definition_embedding = model.encode(definition)
        # concatenate to one sentence embedding
        concat_name_def_query = np.concatenate(
            (definition_embedding, name_embedding), axis=0
        )
        concat_name_def_query = concat_name_def_query.tolist()
        queries = [concat_name_def_query]

        # Query with Semantic Search, k-nearest-neighbor
        # Chroma uses hnswlib https://github.com/nmslib/hnswlib with L2 distance as metric
        # see: https://github.com/chroma-core/chroma/blob/4463d13f951a4d28ade1f7e777d07302ff09069b/chromadb/db/index/hnswlib.py -> suche nach l2

        all_items = collection.count()
        all_items = all_items - 2
        print(all_items)
        # See whether metadata category is in database
        try:
            with_metadata = collection.query(
                query_embeddings=queries,
                #n_results=number_elements,
                n_results = all_items,
                where={"Metalabel": metadata},
            )
        except Exception:
            with_metadata = "Nix"
        # Get best matching result from chroma db (1 for search in different AAS, 1 to many for search in one AAS)
        without_metadata = collection.query(
            query_embeddings=queries,
            #n_results=number_elements,
            n_results = all_items
        )

        # if no matched metadta category -> nlp without metadata
        if with_metadata == "Nix":
            worst_result = without_metadata['distances'][0]
            highest_distance = worst_result[-1]
            print(highest_distance)
            result = without_metadata
            result["matching_method"] = "NLP without Metadata"
            result["matching_algorithm"] = "Semantic search, k-nearest-neighbor with squared L2 distance (euclidean distance), with model gart-labor/eng-distilBERT-se-eclass"

        # if metadata category in database, compare which best result from with and without metadata
        # has lowest distance
        elif with_metadata != "Nix":
            distance_with_meta = with_metadata["distances"][0][0]
            distance_without_meta = without_metadata["distances"][0][0]

            if distance_without_meta <= distance_with_meta:
                worst_result = without_metadata['distances'][0]
                highest_distance = worst_result[-1]
                print(highest_distance)
                result = without_metadata
                result["matching_method"] = "NLP without Metadata"
                result["matching_algorithm"] = "Semantic search, k-nearest-neighbor with squared L2 distance (euclidean distance), with model gart-labor/eng-distilBERT-se-eclass"

            else:
                worst_result = with_metadata['distances'][0]
                highest_distance = worst_result[-1]
                print(highest_distance)
                result = with_metadata
                result["matching_method"] = "NLP with Metadata"
                result["matching_algorithm"] = "Semantic search, k-nearest-neighbor with squared L2 distance (euclidean distance), with model gart-labor/eng-distilBERT-se-eclass and classifier with metadata (unit and or datatype)"

        value = result['documents'][0][0]
        value_dict = json.loads(value)
        #print(result)
        # Old structure for results (if we want to use things like the matched submodel element later on)
        final_results = []
        print(number_elements)
        for i in range(0, number_elements):
            normalized_distance = result['distances'][0][i]/highest_distance
            print(normalized_distance)
            rounded_distance = round(normalized_distance, 2)
            print(rounded_distance)
            value = result['documents'][0][i]
            value_dict = json.loads(value)
            final_result = {
                "matching_method": result['matching_method'],
                "matching_algorithm": result["matching_algorithm"],
                #"matching_distance": result['distances'][0][i],
                "matching_distance": rounded_distance,
                "aas_id": result['metadatas'][0][i]['AASId'],
                "aas_id_short": result['metadatas'][0][i]['AASIdShort'],
                "submodel_id_short": result['metadatas'][0][i]['SubmodelName'],
                "submodel_id": result['metadatas'][0][i]['SubmodelId'],
                "id_short_path": result['metadatas'][0][i]['IdShortPath'],
                "matched_object": value_dict
            }
            #final_result = json.dumps(final_result, indent = 4)
            final_results.append(final_result)
            # test
        
        return final_results

def classify_elements(collections, client_chroma, preferred_name, definition, unit, datatype, semantic_id, number_elements):
    # Load NLP Model
    model = SentenceTransformer("gart-labor/eng-distilBERT-se-eclass")

    with open("metadata.pickle", "rb") as handle:
        metalabel = pickle.load(handle)
    
    results = []
    # Get best matching submodel element from every aas
    for collection in collections:
        collection = client_chroma.get_collection(collection.name)
        result_list = query_aas(model, metalabel, preferred_name, definition, unit, datatype, collection, number_elements)
        result = result_list[0]
        results.append(result)
    
    return results

def classify_elements_one_aas(collection, client_chroma, preferred_name, definition, unit, datatype, semantic_id, number_elements):

    model = SentenceTransformer("gart-labor/eng-distilBERT-se-eclass")

    with open("metadata.pickle", "rb") as handle:
        metalabel = pickle.load(handle)
    # Get reuslts from the one requested aas
    results = query_aas(model, metalabel, preferred_name, definition, unit, datatype, collection, number_elements)
    
    return results

def get_right_collection(collections, aas_id): 
        right_collection = []
        # search for right aas id in all aas collections
        for collection in collections:
            try_collection = collection.get(where={'AASId': aas_id})
            try:
                collection_aas_id = try_collection['metadatas'][0]['AASId']
                right_collection.append(collection)
            except:
                print('Nix')
        if(right_collection == []):
            right_collection = ['NotFound']

        return right_collection