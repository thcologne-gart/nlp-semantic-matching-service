import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import basyx.aas.model
import json
import re
import requests
import ast
from functools import reduce 
import operator

from idta_submodels_information import select_idta_submodels

def create_idta_submodel(data_dict, model, client, submodel_name, set_threshold): 
    #model = SentenceTransformer("gart-labor/eng-distilBERT-se-eclass")
    #model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    #with open(data, 'r') as file:
    #    data_dict = json.load(file)

    aas_submodel_information, submodel_id_for_idta = select_idta_submodels(submodel_name)
    
    aas = data_dict["assetAdministrationShells"]
    aas_id = aas[0]["identification"]["id"]
    aas_submodels = aas[0]["submodels"]
    submodels_ids = []

    for submodel in aas_submodels:
        submodels_ids.append(submodel["keys"][0]["value"])

    submodels = data_dict["submodels"]
    conceptDescriptions = data_dict["conceptDescriptions"]


    # Concept Descriptions vom IDTA!!!! in anderer python hinzufügen


    data = json.dumps(data_dict)
    aas_df = read_submodel(
        data, submodels, conceptDescriptions, submodels_ids 
    )

    aas_df = encode(aas_df, model)

    #metadata, aas_index_str, se_content, se_embedding_name_definition = convert_to_list(
    #    aas_df
    #)

    #client = chromadb.Client(Settings(chroma_api_impl="rest",
    #    chroma_server_host="3.125.157.194",
    #    chroma_server_http_port=8000)
    #)

    #data_dict["submodels"].append(submodel_information)

    data_dict["assetAdministrationShells"][0]["submodels"].append(aas_submodel_information)

    #print(data_dict["submodels"])
    #print(data_dict["assetAdministrationShells"][0]["submodels"])

    aas_with_idta, report = get_chroma_collection(aas_df, data_dict, client, submodel_name, submodel_id_for_idta, set_threshold)

    return aas_with_idta, report

    #created_idta_submodel = create_submodel_basyx(aas_df, data_dict, aas_id)

    """
    client = chromadb.Client(Settings(chroma_api_impl="rest",
        chroma_server_host="18.196.90.82",
        chroma_server_http_port=8000)
    )
    print('hallo')
    model = SentenceTransformer("gart-labor/eng-distilBERT-se-eclass")

    preferred_name = 'producer name'
    definition = 'name of the producer'

    name_embedding = model.encode(preferred_name)
    definition_embedding = model.encode(definition)

    concat_name_def_query = np.concatenate(
        (definition_embedding, name_embedding), axis=0
    )
    concat_name_def_query = concat_name_def_query.tolist()
    queries = [concat_name_def_query]
    
    collection = client.get_collection(name="aas_digital_nameplate")
    print(collection)
    without_metadata = collection.query(
        query_embeddings=queries,
        n_results=3
    )
    print(without_metadata)
    """
    

def read_submodel(data, submodels, conceptDescriptions, submodels_ids ):
    df = pd.DataFrame(
        columns=[
            "SubmodelId",
            "SubmodelName",
            "SubmodelSemanticId",
            "SEContent",
            "SESemanticId",
            "SEModelType",
            "SEIdShort",
            "SEValue",
            "Definition",
            "PreferredName",
            "Datatype",
            "Unit",
            "IdShortPath"
        ]
    )
    # Aufbereiten aller Concept descriptions als pandas dataframe, damit diese nachher einfacher untersucht werden können
    df_cd = prepare_cd(conceptDescriptions)
    # Auslesen der Teilmodelle
    for submodel in submodels:
        submodel_name = submodel["idShort"]
        submodel_id = submodel["identification"]["id"]
        # Muss gemacht werden, da Anzahl der Teilmodelle innerhalb der AAS und des Env nicht immer übereisntimmen
        if submodel_id in submodels_ids:
            semantic_id_existing = submodel["semanticId"]["keys"]
            if not semantic_id_existing:
                submodel_semantic_id = "Not defined"
            else:
                submodel_semantic_id = semantic_id_existing[0]["value"]
            submodel_elements = submodel["submodelElements"]
            # Auslesen Submodel Elements
            for submodel_element in submodel_elements:
                id_short_path = submodel_name
                content = []
                content.append(submodel_element)

                (
                    se_type,
                    se_semantic_id,
                    se_semantic_id_local,
                    se_id_short,
                    value,
                    id_short_path,
                    se_description
                ) = get_values(submodel_element, id_short_path)

                # When Concept Description local dann auslesen der Concept Description
                if se_semantic_id_local == True:
                    cd_content = get_concept_description(se_semantic_id, df_cd, se_id_short, se_description)
                    definition = cd_content["Definition"]
                    preferred_name = cd_content["PreferredName"]
                    datatype = cd_content["Datatype"]
                    unit = cd_content["Unit"]

                else:
                    definition = se_description
                    preferred_name = se_id_short
                    datatype = "NaN"
                    unit = "NaN"
                new_row = pd.DataFrame(
                    {
                        "SubmodelId": submodel_id,
                        "SubmodelName": submodel_name,
                        "SubmodelSemanticId": submodel_semantic_id,
                        "SEContent": content,
                        "SESemanticId": se_semantic_id,
                        "SEModelType": se_type,
                        "SEIdShort": se_id_short,
                        "SEValue": value,
                        "Definition": definition,
                        "PreferredName": preferred_name,
                        "Datatype": datatype,
                        "Unit": unit,
                        "IdShortPath": id_short_path
                    }
                )
                #print(new_row.loc[0, 'Definition'])
                #print(new_row.loc[0, 'PreferredName'])
                df = pd.concat([df, new_row], ignore_index=True)

                # Wenn Submodel Element Collection dann diese Werte auch auslesen
                if se_type == "SubmodelElementCollection":
                    df = get_values_sec(
                        data,
                        df_cd,
                        content,
                        df,
                        submodel_id,
                        submodel_name,
                        submodel_semantic_id,
                        id_short_path
                    )
        else:
            continue

    return df

def prepare_cd(conceptDescriptions):
    df_cd = pd.DataFrame(
        columns=["SemanticId", "Definition", "PreferredName", "Datatype", "Unit"]
    )
    # In den leeren DF werden alle Concept Descriptions eingelesen
    for cd in conceptDescriptions:
        semantic_id = cd["identification"]["id"]
        data_spec = cd["embeddedDataSpecifications"][0]["dataSpecificationContent"]
        preferred_name = data_spec["preferredName"]
        short_name = data_spec["shortName"]
        if len(preferred_name) > 1:
            for name_variant in preferred_name:
                if (
                    name_variant["language"] == "EN"
                    or name_variant["language"] == "en"
                    or name_variant["language"] == "EN?"
                ):
                    name = name_variant["text"]
        elif len(preferred_name) == 1:
            name = preferred_name[0]["text"]
        elif len(preferred_name) == 0:
            short_name = data_spec["shortName"]
            if len(short_name) == 0:
                name = "NaN"
            else:
                name = short_name[0]["text"]

        definition = data_spec["definition"]
        if len(definition) > 1:
            for definition_variant in definition:
                if (
                    definition_variant["language"] == "EN"
                    or definition_variant["language"] == "en"
                    or definition_variant["language"] == "EN?"
                ):
                    chosen_def = definition_variant["text"]
        elif len(definition) == 1:
            chosen_def = definition[0]["text"]
        elif len(definition) == 0:
            chosen_def = "NaN"

        if data_spec["dataType"] == "":
            datatype = "NaN"
        else:
            datatype = data_spec["dataType"]

        if data_spec["unit"] == "":
            unit = "NaN"
        else:
            unit = data_spec["unit"]

        new_entry = pd.DataFrame(
            {
                "SemanticId": semantic_id,
                "Definition": chosen_def,
                "PreferredName": name,
                "Datatype": datatype,
                "Unit": unit,
            },
            index=[0],
        )
        df_cd = pd.concat([df_cd, new_entry], ignore_index=True)
    return df_cd

def get_values(submodel_element, id_short_path):
    # Auslesen der Submodel Element Werte
    #print(submodel_element)
    se_type = submodel_element["modelType"]["name"]
    se_semantic_id = submodel_element["semanticId"]["keys"][0]["value"]
    se_semantic_id_local = submodel_element["semanticId"]["keys"][0]["local"]
    se_id_short = submodel_element["idShort"]
    id_short_path = id_short_path + '.' + se_id_short
    value = []
    if se_type == 'MultiLanguageProperty':
        #print(submodel_element)
        se_value = submodel_element["value"]["langString"][0]["text"]
    else:
        se_value = submodel_element["value"]
    value.append(se_value)

    if 'descriptions' in submodel_element:
        definition = submodel_element["descriptions"]
        if len(definition) > 1:
            for definition_variant in definition:
                if (
                    definition_variant["language"] == "EN"
                    or definition_variant["language"] == "en"
                    or definition_variant["language"] == "EN?"
                ):
                    chosen_def = definition_variant["text"]
        elif len(definition) == 1:
            chosen_def = definition[0]["text"]
        elif len(definition) == 0:
            chosen_def = "NaN"
    else: 
        chosen_def = 'NaN'
    
    #print(chosen_def)

    return se_type, se_semantic_id, se_semantic_id_local, se_id_short, value, id_short_path, chosen_def

def get_concept_description(semantic_id, df_cd, se_id_short, se_description):
    """
    Retrieve the concept description for a given semantic id.

    Parameters
    ----------
    semantic_id : str
        The semantic id for the concept.
    df_cd : pandas.DataFrame
        The concept description dataframe.

    Returns
    -------
    pandas.DataFrame
        The concept description.
    """

    cd_content = df_cd.loc[df_cd["SemanticId"] == semantic_id]

    if cd_content.empty:
        cd_content = pd.DataFrame(
            {
                "SemanticId": semantic_id,
                "Definition": se_description,
                "PreferredName": se_id_short,
                "Datatype": "NaN",
                "Unit": "NaN",
            },
            index=[0],
        )
    #print(cd_content)
    if cd_content.iloc[0]['Definition'] == 'NaN':
        #cd_content.at[0, 'Definition'] = se_description
        cd_content_copy = cd_content.iloc[[0]].copy()
        cd_content_copy['Definition'] = se_description
        cd_content.iloc[[0]] = cd_content_copy

    if cd_content.iloc[0]['PreferredName'] == 'NaN':
        #cd_content.at[0, 'PreferredName'] = se_id_short
        cd_content_copy = cd_content.iloc[[0]].copy()
        cd_content_copy['PreferredName'] = se_id_short
        cd_content.iloc[[0]] = cd_content_copy
    cd_content = cd_content.iloc[0]
    #print(cd_content)

    return cd_content

def get_values_sec(
    data,
    df_cd,
    content,
    df,
    submodel_id,
    submodel_name,
    submodel_semantic_id,
    id_short_path
):
    collection_values = content[0]["value"]
    for element in collection_values:
        sec_id_short_path = id_short_path
        content = []
        content.append(element)

        se_type, se_semantic_id, se_semantic_id_local, se_id_short, value, sec_id_short_path, se_description = get_values(
            element, sec_id_short_path
        )
        if se_type == "SubmodelElementCollection":
            if se_semantic_id_local == True:
                cd_content = get_concept_description(se_semantic_id, df_cd, se_id_short, se_description)
                definition = cd_content["Definition"]
                preferred_name = cd_content["PreferredName"]
                datatype = cd_content["Datatype"]
                unit = cd_content["Unit"]

            else:
                definition = se_description
                preferred_name = se_id_short
                datatype = "NaN"
                unit = "NaN"

            new_row = pd.DataFrame(
                {
                    "SubmodelId": submodel_id,
                    "SubmodelName": submodel_name,
                    "SubmodelSemanticId": submodel_semantic_id,
                    "SEContent": content,
                    "SESemanticId": se_semantic_id,
                    "SEModelType": se_type,
                    "SEIdShort": se_id_short,
                    "SEValue": value,
                    "Definition": definition,
                    "PreferredName": preferred_name,
                    "Datatype": datatype,
                    "Unit": unit,
                    "IdShortPath": sec_id_short_path,
                }
            )
            df = pd.concat([df, new_row], ignore_index=True)

            content = []
            content.append(element)
            # Rekursive Funktion -> so oft durchlaufen bis unterste Ebene der Collections erreicht ist, so werden verschachteltet SECs bis zum Ende ausgelesen
            df = get_values_sec(
                data,
                df_cd,
                content,
                df,
                submodel_id,
                submodel_name,
                submodel_semantic_id,
                sec_id_short_path
            )

        else:
            if se_semantic_id_local == True:
                cd_content = get_concept_description(se_semantic_id, df_cd, se_id_short, se_description)
                definition = cd_content["Definition"]
                preferred_name = cd_content["PreferredName"]
                datatype = cd_content["Datatype"]
                unit = cd_content["Unit"]

            else:
                definition = "NaN"
                preferred_name = "NaN"
                datatype = "NaN"
                unit = "NaN"

            new_row = pd.DataFrame(
                {
                    "SubmodelId": submodel_id,
                    "SubmodelName": submodel_name,
                    "SubmodelSemanticId": submodel_semantic_id,
                    "SEContent": content,
                    "SESemanticId": se_semantic_id,
                    "SEModelType": se_type,
                    "SEIdShort": se_id_short,
                    "SEValue": value,
                    "Definition": definition,
                    "PreferredName": preferred_name,
                    "Datatype": datatype,
                    "Unit": unit,
                    "IdShortPath": sec_id_short_path
                }
            )
            df = pd.concat([df, new_row], ignore_index=True)

    return df

def add_spaces(name):
    # Use regex to match camel case pattern
    pattern = r'(?<=[a-z])(?=[A-Z])'
    # Insert space between matched pattern
    replaced_string = re.sub(pattern, ' ', name)
    return replaced_string

def encode(aas_df, model):
    # Einsatz von Sentence Bert um Embeddings zu kreieren
    aas_df['PreferredName'] = aas_df['PreferredName'].apply(add_spaces)
    aas_df["PreferredName"] = "Name: " + aas_df["PreferredName"].astype(str)
    aas_df["Definition"] = "Description: " + aas_df["Definition"].astype(str) + "; "
    corpus_names = aas_df.loc[:, "PreferredName"]
    corpus_definitions = aas_df.loc[:, "Definition"]
    embeddings_definitions = model.encode(corpus_definitions, show_progress_bar=True)
    embeddings_names = model.encode(corpus_names, show_progress_bar=True)
    concat_name_def_emb = np.concatenate(
        (embeddings_definitions, embeddings_names), axis=1
    )
    aas_df["EmbeddingNameDefinition"] = concat_name_def_emb.tolist()
    return aas_df


def convert_to_list(aas_df):
    # Für die Datenbank werden teilweise Listen gebraucht
    aas_index = aas_df.index.tolist()
    aas_index_str = [str(r) for r in aas_index]
    se_content = aas_df["SEContent"].tolist()
    se_embedding_name_definition = aas_df["EmbeddingNameDefinition"].tolist()

    aas_df_dropped = aas_df.drop(
        ["EmbeddingNameDefinition", "SEContent", "SEValue"], axis=1
    )

    metadata = aas_df_dropped.to_dict("records")

    return metadata, aas_index_str, se_content, se_embedding_name_definition

def get_chroma_collection(aas_df, data_dict, client, submodel_name, submodel_id_for_idta, set_threshold):

    aas_df['Distance'] = np.nan
    aas_df['MatchedSESemanticId'] = np.nan
    aas_df['MatchedPreferredName'] = np.nan
    aas_df['MatchedDefinition'] = np.nan
    aas_df['MatchedIdShortPath'] = np.nan
    aas_df['MatchedSEContent'] = np.nan
    aas_df['MatchedSEIdShort'] = np.nan
    aas_df['MatchedIdShortPathValues'] = np.nan
    
    collection = client.get_collection(name="aas_digital_nameplate")
    
    embeddings_list = list(aas_df['EmbeddingNameDefinition'])
    chroma_result = collection.query(
        query_embeddings = embeddings_list,
        n_results=1
    )


    idta_submodel_content = chroma_result['metadatas'][0][0]['SubmodelContent']
    idta_submodel_dictionary = json.loads(idta_submodel_content)  
    idta_submodel_dictionary["identification"]["id"] = submodel_id_for_idta
    data_dict['submodels'].append(idta_submodel_dictionary)
    number_submodels = len(data_dict['submodels']) - 1

    #paths = collection.get()
    #print(paths)
    #all_submodel_elements_numbers = []
    #for result in paths['metadatas']:
        #print(result['IdShortPathValues'])
        #value = ast.literal_eval(result['IdShortPathValues'])
        #all_submodel_elements_numbers.append(value)


    mapped_submodel_elements = []
    report = []
    #print(len(aas_df))
    for index, row in aas_df.iterrows(): 
        try:
            result = collection.get(where={"SESemanticId": row['SESemanticId']})
            
            #print(result['metadatas'][0]['PreferredName'])
            #print(row['PreferredName'])
            aas_df.at[index, 'Distance'] = 0
            aas_df.at[index, 'MatchedDefinition'] = result['metadatas'][0]['Definition']
            matched_id_short_path_values = result['metadatas'][0]['IdShortPathValues']
            aas_df.at[index, 'MatchedIdShortPathValues'] = matched_id_short_path_values
            matched_id_short_path_values = ast.literal_eval(matched_id_short_path_values)
            matched_preferred_name = result['metadatas'][0]['PreferredName']
            aas_df.at[index, 'MatchedPreferredName'] = matched_preferred_name
            matched_semantic_id = result['metadatas'][0]['SESemanticId']
            aas_df.at[index, 'MatchedSESemanticId'] = matched_semantic_id
            id_short_path = result['metadatas'][0]['IdShortPath']
            aas_df.at[index, 'MatchedIdShortPath'] = id_short_path
            matched_se_id_short = result['metadatas'][0]['SEIdShort']
            aas_df.at[index, 'MatchedSEIdShort'] = matched_se_id_short
            matched_se_model_type = result['metadatas'][0]['SEModelType']
            matched_definition = result['metadatas'][0]['Definition']

            #print(aas_df.at[index, 'Distance'])
            #print('--------------------------')

            report.append(
                {
                   'RequestedSubmodelElement':  
                    {
                        'Name': row['PreferredName'],
                        'Definition': row['Definition'],
                        'SemanticId': row['SESemanticId']
                    },
                    'IDTASubmodelElement': 
                    {
                        'Name': matched_preferred_name,
                        'Definition': matched_definition,
                        'SemanticId': matched_semantic_id
                    },
                    'MattchingInformation': {
                        "MatchingMethod": "Semantic equivalent, same semantic Id",
                        "MatchingAlgorithm": "None",
                        "MatchingDistance": 0,
                        "Accepted": True
                    }
                }
            )

            #id_short_path_splitted = id_short_path.split('.')

        except Exception:

            #print(chroma_result['metadatas'][index][0]['PreferredName'])
            #print(row['PreferredName'])    
            distance = chroma_result['distances'][index][0]
            aas_df.at[index, 'Distance'] = distance
            aas_df.at[index, 'MatchedDefinition'] = chroma_result['metadatas'][index][0]['Definition']
            matched_id_short_path_values = chroma_result['metadatas'][index][0]['IdShortPathValues']
            aas_df.at[index, 'MatchedIdShortPathValues'] = matched_id_short_path_values
            matched_id_short_path_values = ast.literal_eval(matched_id_short_path_values)
            matched_preferred_name = chroma_result['metadatas'][index][0]['PreferredName']
            aas_df.at[index, 'MatchedPreferredName'] = matched_preferred_name
            matched_semantic_id = chroma_result['metadatas'][index][0]['SESemanticId']
            aas_df.at[index, 'MatchedSESemanticId'] = matched_semantic_id
            id_short_path = chroma_result['metadatas'][index][0]['IdShortPath']
            aas_df.at[index, 'MatchedIdShortPath'] = id_short_path
            matched_se_id_short = chroma_result['metadatas'][index][0]['SEIdShort']
            aas_df.at[index, 'MatchedSEIdShort'] = matched_se_id_short
            matched_se_model_type = chroma_result['metadatas'][index][0]['SEModelType']
            matched_definition = chroma_result['metadatas'][index][0]['Definition']
            #id_short_path_splitted = id_short_path.split('.')

            #print(aas_df.at[index, 'Distance'])

            if distance >= set_threshold:
                aas_df = aas_df.drop(index)
                accepted = False
            else:
                accepted = True
            
            #print(len(aas_df))
            #print('------------------------')

            report.append(
                {
                   'RequestedSubmodelElement':  
                    {
                        'Name': row['PreferredName'],
                        'Definition': row['Definition'],
                        'SemanticId': row['SESemanticId']
                    },
                    'IDTASubmodelElement': 
                    {
                        'Name': matched_preferred_name,
                        'Definition': matched_definition,
                        'SemanticId': matched_semantic_id
                    },
                    'MattchingInformation': {
                        "MatchingMethod": "NLP with metadata",
                        "MatchingAlgorithm": "Semantic search, k-nearest-neighbor with cosine distance, with model gart-labor/eng-distilBERT-se-eclass",
                        "MatchingDistance": distance,
                        "Accepted": accepted
                    }
                }
            )

        if len (matched_id_short_path_values) == 1:
            submodel_element_content = chroma_result['documents'][index][0]
            submodel_element_content = json.loads(submodel_element_content)
            #print(type(submodel_element_content))

            if matched_se_model_type == 'MultiLanguageProperty':
                
                
                if matched_id_short_path_values not in mapped_submodel_elements:
                    mapped_submodel_elements.append(matched_id_short_path_values)
                    data_dict['submodels'][number_submodels]['submodelElements'][matched_id_short_path_values[0]]["value"]["langString"][0]["text"] = row["SEValue"]

            elif matched_se_model_type == 'Property':
                
                if matched_id_short_path_values not in mapped_submodel_elements:
                    mapped_submodel_elements.append(matched_id_short_path_values)
                    data_dict['submodels'][number_submodels]['submodelElements'][matched_id_short_path_values[0]]["value"] = row["SEValue"]


        elif len(matched_id_short_path_values) > 1:
            
            if (matched_se_model_type == 'Property' or matched_se_model_type == 'MultiLanguageProperty') and (row['SEModelType'] == 'Property' or row['SEModelType'] == 'MultiLanguageProperty'):
                
                appended_list = [item for sublist in [[x, 'value'] for x in matched_id_short_path_values] for item in sublist]

                if matched_se_model_type == 'MultiLanguageProperty':
                    extra_elements = ["langString", 0, "text"]
                    appended_list.extend(extra_elements)        

                value = row["SEValue"]

                set_by_path(data_dict['submodels'][number_submodels]['submodelElements'], appended_list, value)

    """
    with open('final_aas.json', 'w') as file:
        #json.dump(all_classified_datapoint_collections, file)
        #json.dump(json_aas, file)
        #file.write(json_aas)
        json.dump(data_dict, file, indent=2)
    """
    with open('matching_report.json', 'w') as report_file:
        #json.dump(all_classified_datapoint_collections, file)
        #json.dump(json_aas, file)
        #file.write(json_aas)
        json.dump(report, report_file, indent=2)
    
    
    #aas_with_idta = json.dumps(data_dict, indent=2)
    
    return data_dict, report

def get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)

def set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    get_by_path(root, items[:-1])[items[-1]] = value


def create_submodel_basyx(aas_df, data_dict, aas_id):

    #aas_server = "http://3.83.126.51:4001/aasServer/shells/AAS_Digital_Nameplate/aas/submodels/Nameplate/submodel/submodelElements"
    #aas_server = "http://3.83.126.51:4001/aasServer/shells/AAS_Digital_Nameplate/aas/submodels/Nameplate/submodel"
    aas_server = "http://3.83.126.51:4001/aasServer/shells"
    aas_template_nameplate = '/AAS_Digital_Nameplate/aas'

    aas = data_dict['assetAdministrationShells'][0]
    aas['asset']['modelType'] = {'name': 'Asset'}
    aas['asset']['dataSpecification'] = [],
    aas['asset']['identification'] = {'idType': 'IRDI', 'id': ''}
    aas['asset']['kind'] = 'Instance'
    aas['asset']['embeddedDataSpecifications'] = []
    aas['asset']['idShort'] = ''
    submodel_id = 'www.example.com/ids/sm/1225_9020_5022_1974'
    submodel_idta_nameplate = {
        'keys': [{'type': 'Submodel', 'local': True, 'value': submodel_id, 'index': 0, 'idType': 'IRI'}]
    }
    aas['submodels'].append(submodel_idta_nameplate)

    #data_dict['assetAdministrationShells'][0]['submodels'].append(submodel_idta_nameplate)
    #print(aas)

    aas_string = json.dumps(aas)
    
    new_aas_id = aas_id.replace('/', '%2F')
    print(new_aas_id)
    put_aas = aas_server + '/' + new_aas_id


    deleted = requests.delete(put_aas)

    
    print(put_aas)
    headers = {"Content-Type": "application/json"}
    response = requests.put(put_aas, data = aas_string, headers = headers)
    print(response.content)
    submodels = data_dict['submodels']
    #print(submodels)
    print(len(submodels))
    for submodel in submodels:
        submodel_id_short = submodel['idShort']
        print(submodel_id_short)
        submodel_string = json.dumps(submodel)
        #print(submodel_string)
        put_submodel = put_aas + '/aas/submodels/' + submodel_id_short
        print(put_submodel)
        response_submodel = requests.put(put_submodel, data = submodel_string, headers = headers)

    final_idta_submodel = {
        "semanticId": {
            "keys": [
            {
                "type": "ConceptDescription",
                "local": "true",
                "value": "https://admin-shell.io/zvei/nameplate/2/0/Nameplate",
                "index": 0,
                "idType": "IRI"
            }
            ]
        },
        "qualifiers": [],
        "hasDataSpecification": [],
        "identification": {
            "idType": "IRI",
            "id": "www.example.com/ids/sm/1225_9020_5022_1974"
        },
        "idShort": "Nameplate_IDTA",
        "modelType": {
            "name": "Submodel"
        },
        "kind": "Instance",
        "submodelElements": []
    }

    idta_nameplate_id = "http://3.83.126.51:4001/aasServer/shells/AAS_Digital_Nameplate/aas/submodels/Nameplate/submodel"
    response_nameplate = requests.get(idta_nameplate_id)
    response_text = response_nameplate.text
    #print(response_text)
    encoded_text = response_text.encode('utf-8')

    response_dict = json.loads(encoded_text)
    submodel_id_short = 'Nameplate_IDTA'
    response_dict['idShort'] = submodel_id_short
    #print(response_dict)
    submodel_string_nameplate = json.dumps(response_dict)

    put_submodel_idta_nameplate = put_aas + '/aas/submodels/' + submodel_id_short
    print(put_submodel_idta_nameplate)
    response_submodel_nameplate = requests.put(put_submodel_idta_nameplate, data = submodel_string_nameplate, headers = headers)
    
    for index, row in aas_df.iterrows():
        id_short_path = row['MatchedIdShortPath']
        id_short_splitted = id_short_path.split('.', 1)[1]
        id_short_path_replaced = id_short_splitted.replace('.', '/')
        basyx_path = put_submodel_idta_nameplate + '/submodel/submodelElements/' + id_short_path_replaced
        print(basyx_path)
        response_se_element = requests.get(basyx_path)
        response_dict = json.loads(response_se_element.text)
        se_model_type = response_dict['modelType']['name']
        print(se_model_type)
        if se_model_type == 'Property':
            response_dict['value'] = row['SEValue']
            #print(response_dict)
            se_string = json.dumps(response_dict)
            request= requests.put(basyx_path, data = se_string, headers = headers)
            del response_dict['parent']
            final_idta_submodel['submodelElements'].append(response_dict)
            #print(request)

        elif se_model_type == 'MultiLanguageProperty':
            print(row)
            response_dict["value"]["langString"][0]["text"] = row['SEValue']
            print(response_dict)
            se_string = json.dumps(response_dict)
            request= requests.put(basyx_path, data = se_string, headers = headers)
            del response_dict['parent']
            final_idta_submodel['submodelElements'].append(response_dict)
            print(request)

    get_aas_with_submodels = put_aas + '/aas/submodels/Nameplate_IDTA/submodel'
    final_submodel = requests.get(get_aas_with_submodels)
    final_submodel = final_submodel.text
    # Umformen zu wohlformiertem JSON
    #encoded_text = response_text.encode('utf-8')
    json_submodel = json.loads(final_submodel)
    print(type(json_submodel))
    #print(json_submodel)


    #data_dict['submodels'].append(json_submodel)
    data_dict['submodels'].append(final_idta_submodel)
    #json_aas = json.dumps(final_aas, indent=2)

    with open('final_aas.json', 'w') as file:
        #json.dump(all_classified_datapoint_collections, file)
        #json.dump(json_aas, file)
        #file.write(json_aas)
        json.dump(data_dict, file, indent=2)
    

    # Wie bekomme ich die Inhalt der Submodels da rein, so nur leere AAS aus Basyx
    # Was ist mit Submodel Elememnt Collections mapping? wie geht das?
    # Warum keine werte bei MultiLanguageProperty?
    # Chroma Instanzen für normale VWS als auch auch IDTA sind runter gefahren

            


    #response_dict = json.loads(response_text)
    #new_aas_id = 
    #print(type(response_dict))
    #print(response_dict['submodelElements'])
    #print(len(response_dict['submodelElements']))

"""
if __name__ == "__main__":
    set_threshold = 0.5
    submodel_name = 'https://admin-shell.io/zvei/nameplate/2/0/Nameplate'
    data = test_chroma("test_submodels/14_Siemens_aas.json", submodel_name, set_threshold)
    #data = test_chroma("test_submodels/08_SchneiderElectric.json")

"""
    