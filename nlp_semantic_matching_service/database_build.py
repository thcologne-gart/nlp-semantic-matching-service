from sentence_transformers import SentenceTransformer, util
import json
import time
import pandas as pd
import numpy as np
import torch
import pickle

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
#from chromadb.db.clickhouse import NoDatapointsException

import requests

import os
import openai
openai.api_key = ""

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
        print(submodel_element)
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
    
    print(chosen_def)

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
    print(cd_content)
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
    print(cd_content)

    return cd_content


def get_values_sec(
    data,
    df_cd,
    content,
    df,
    aas_id,
    aas_name,
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
                    "AASContent": data,
                    "AASId": aas_id,
                    "AASIdShort": aas_name,
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
                aas_id,
                aas_name,
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
                    "AASContent": data,
                    "AASId": aas_id,
                    "AASIdShort": aas_name,
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


def set_up_metadata(metalabel, df):
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

    dataset = df
    dataset["unit_lowercase"] = dataset["Unit"]
    dataset["unit_lowercase"] = dataset["unit_lowercase"].str.lower()
    dataset["unit_categ"] = dataset["unit_lowercase"].map(unit_mapping)

    dataset["datatype_lowercase"] = dataset["Datatype"]
    dataset["datatype_lowercase"] = dataset["datatype_lowercase"].str.lower()
    dataset["datatype_categ"] = dataset["datatype_lowercase"].map(datatype_mapping)

    dataset = dataset.fillna("NaN")
    dataset["index"] = dataset.index

    # uni_datatype=dataset['datatype_categ'].unique()
    # uni_unit=dataset['unit_categ'].unique()
    unique_labels_set = set()

    dataset["Metalabel"] = ""
    for i in range(0, len(dataset["Metalabel"])):
        concat = (str(dataset["unit_categ"][i]), str(dataset["datatype_categ"][i]))
        keys = [k for k, v in metalabel.items() if v == concat]
        dataset["Metalabel"][i] = keys[0]
        unique_labels_set.add(keys[0])
    unique_label = list(unique_labels_set)
    #print(unique_label)

    return dataset


def encode(aas_df, model):
    # Einsatz von Sentence Bert um Embeddings zu kreieren
    aas_df["PreferredName"] = "Name: " + aas_df["PreferredName"].astype(str)
    aas_df["Definition"] = "Description: " + aas_df["Definition"].astype(str) + "; "
    corpus_names = aas_df.loc[:, "PreferredName"]
    corpus_definitions = aas_df.loc[:, "Definition"]
    embeddings_definitions = model.encode(corpus_definitions, show_progress_bar=True)
    embeddings_names = model.encode(corpus_names, show_progress_bar=True)
    concat_name_def_emb = np.concatenate(
        (embeddings_definitions, embeddings_names), axis=1
    )
    # aas_df['EmbeddingDefinition'] = embeddings_definitions.tolist()
    # aas_df['EmbeddingName'] = embeddings_names.tolist()
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

def set_up_chroma(
    metadata, aas_index_str, se_content, se_embedding_name_definition, aas_name, client
):
    aas_name = aas_name.lower()
    # Kein Großbuchstaben in Datenbank erlaubt
    print(aas_name)
    # client = chromadb.Client(Settings(
    #    chroma_db_impl="duckdb+parquet",
    #    persist_directory="./drive/My Drive/Colab/NLP/SemantischeInteroperabilität/Deployment" # Optional, defaults to .chromadb/ in the current directory
    # ))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="gart-labor/eng-distilBERT-se-eclass"
    )
    collection = client.get_or_create_collection(
        name=aas_name, embedding_function=emb_fn, metadata={"hnsw:space": "cosine"}
    )
    print(client.list_collections())

    aas_content_string = []
    # Umwandeln in Json damit es in db geschrieben werden kann
    for element in se_content:
        content = json.dumps(element)
        aas_content_string.append(content)
    
    print(client.heartbeat())
    print(collection)
    print(collection.count())
    items = collection.count()  # returns the number of items in the collection
    #print(collection)
    print("Datenbank erstellt, Anzahl Items:")
    print(items)
    if items == 0: 
        # Hinzufügen der SE Inhalte, der Embeddings und weiterer Metadaten in collection der Datenbank
        collection.add(
            documents=aas_content_string,
            embeddings=se_embedding_name_definition,
            metadatas=metadata,
            ids=aas_index_str,
        )
        items = collection.count()  # returns the number of items in the collection
        print("------------")
        print("Datenbank befüllt, Anzahl items:")
        print(items)
    else:
        print("-----------")
        print("AAS schon vorhanden")

    return collection

def predict_aas_type(aas_df, model, client_chroma_eclass):
    queries = []
    semantic_id_manufacturer = "0173-1#02-AAO677#002"
    manufacturer_name = "Manufacturer name"
    manufacturer_name_def = "legally valid designation of the natural or judicial person which is directly responsible for the design, production, packaging and labeling of a product in respect to its being brought into circulation"
    mn_embedding = model.encode(manufacturer_name)
    mnd_embedding = model.encode(manufacturer_name_def)
    concat_name_def_mn = np.concatenate(
        (mnd_embedding, mn_embedding), axis=0
    )
    concat_name_def_mn = concat_name_def_mn.tolist()
    dic_manufacturer_name = {'embedding': concat_name_def_mn, 'semantic_id': semantic_id_manufacturer}
    queries.append(dic_manufacturer_name)
                                 
    semantic_id_product_description = "0173-1#02-AAW338#001"
    product_description = "Manufacturer product designation"
    product_description_def = "Short description of the product (short text)"
    pd_embedding = model.encode(product_description)
    pdd_embedding = model.encode(product_description_def)
    concat_name_def_pd = np.concatenate(
        (pdd_embedding, pd_embedding), axis=0
    )
    concat_name_def_pd = concat_name_def_pd.tolist()
    dic_product_description = {'embedding': concat_name_def_pd, 'semantic_id': semantic_id_product_description}
    queries.append(dic_product_description)

    semantic_id_product_family = "0173-1#02-AAU731#001"
    product_family = "Manufacturer product family"
    product_family_def = "2nd level of a 3 level manufacturer specific product hierarchy"
    pf_embedding = model.encode(product_family)
    pfd_embedding = model.encode(product_family_def)
    concat_name_def_pf = np.concatenate(
        (pfd_embedding, pf_embedding), axis=0
    )
    concat_name_def_pf = concat_name_def_pf.tolist()
    dic_product_family = {'embedding': concat_name_def_pf, 'semantic_id': semantic_id_product_family}
    queries.append(dic_product_family)

    for element in queries:
        result = aas_df[aas_df['SESemanticId'] == element['semantic_id']]
        if result.empty:
            print('mit nlp')
            cos_scores = util.cos_sim(element['embedding'], aas_df['EmbeddingNameDefinition'])[0]
            top_result = torch.topk(cos_scores, k= 1)
            print(top_result)
            index = top_result.indices
            index_value = index.item()
            aas_se_value = aas_df.loc[index_value]['SEValue']
            if not aas_se_value:
                aas_se_value = 'not defined'
            if element['semantic_id'] == semantic_id_manufacturer:
                mn_value = aas_se_value
            elif element['semantic_id'] == semantic_id_product_description:
                pd_value = aas_se_value
            elif element['semantic_id'] == semantic_id_product_family:
                pf_value = aas_se_value
        else: 
            aas_se_value = result['SEValue']
            print('ohne nlp')
            aas_se_value = aas_se_value.iloc[0]
            if not aas_se_value:
                aas_se_value = 'not defined'
            if element['semantic_id'] == semantic_id_manufacturer:
                mn_value = aas_se_value
            elif element['semantic_id'] == semantic_id_product_description:
                pd_value = aas_se_value
            elif element['semantic_id'] == semantic_id_product_family:
                pf_value = aas_se_value
    
    aas_id_short = aas_df.loc[0]['AASIdShort'] 
    query = f'{mn_value}, {pd_value}, {pf_value}'
    print(query)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                # Open AI API Key
                api_key="",
                model_name="text-embedding-ada-002"
            )
    print(client_chroma_eclass.list_collections())
    collection = client_chroma_eclass.get_collection(name="eclass_embeddings", embedding_function=openai_ef)
    eclass_category = collection.query(
        query_texts = [query],
        n_results = 1,
    )
    print(eclass_category)
    aas_eclass_type = eclass_category['metadatas'][0][0]['PreferredName']
    aas_eclass_type_irdi = eclass_category['metadatas'][0][0]['IRDI']
    #print(aas_type)
    #print(completion)
    print(aas_eclass_type)
    print(aas_eclass_type_irdi)

    aas_df['AASType'] = aas_eclass_type
    aas_df['AASTypeEclassIrdi'] = aas_eclass_type_irdi
    #print(aas_df.loc[0])

    return aas_df

def read_aas(data, aas, submodels, assets, conceptDescriptions, submodels_ids, metalabel):
    df = pd.DataFrame(
        columns=[
            "AASContent",
            "AASId",
            "AASIdShort",
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
            "IdShortPath",
            #"AASType",
            #"AASTypeEclassIrdi"
        ]
    )
    aas_id = aas[0]["identification"]["id"]
    aas_name = aas[0]["idShort"]
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
                        "AASContent": data,
                        "AASId": aas_id,
                        "AASIdShort": aas_name,
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
                print(new_row.loc[0, 'Definition'])
                print(new_row.loc[0, 'PreferredName'])
                df = pd.concat([df, new_row], ignore_index=True)

                # Wenn Submodel Element Collection dann diese Werte auch auslesen
                if se_type == "SubmodelElementCollection":
                    df = get_values_sec(
                        data,
                        df_cd,
                        content,
                        df,
                        aas_id,
                        aas_name,
                        submodel_id,
                        submodel_name,
                        submodel_semantic_id,
                        id_short_path
                    )
        else:
            continue

    df = set_up_metadata(metalabel, df)

    return df, aas_name, aas_id, aas_name

def post_aas_basyx(aas_file, data, aas_id, aas_name):
    print(type(data))
    data_dict = json.loads(data)
    print(type(data_dict))

    aas = data_dict['assetAdministrationShells'][0]
    print(aas)
    aas['asset']['modelType'] = {'name': 'Asset'}
    aas['asset']['dataSpecification'] = [],
    aas['asset']['identification'] = {'idType': 'IRDI', 'id': ''}
    aas['asset']['kind'] = 'Instance'
    aas['asset']['embeddedDataSpecifications'] = []
    aas['asset']['idShort'] = ''
    print(aas)

    aas_string = json.dumps(aas)
    # AAS Basyx Server
    aas_server = ""
    
    response = requests.get(aas_server)
    print(response)
    new_aas_id = aas_id.replace('/', '%2F')
    print(new_aas_id)
    put_aas = aas_server + '/' + new_aas_id
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
        put_submodel = put_aas + '/aas/submodels/' + submodel_id_short
        print(put_submodel)
        response_submodel = requests.put(put_submodel, data = submodel_string, headers = headers)                

    return response


def index_corpus(aas_file, data, model, metalabel, client_chroma, client_chroma_eclass):
#def index_corpus(aas_file, data, model, metalabel, client_chroma):
    #aas_response = post_aas_basyx(data)
    # Start Punkt
    aas = data["assetAdministrationShells"]
    aas_submodels = aas[0]["submodels"]
    submodels_ids = []
    for submodel in aas_submodels:
        submodels_ids.append(submodel["keys"][0]["value"])
    submodels = data["submodels"]
    conceptDescriptions = data["conceptDescriptions"]
    assets = data["assets"]

    data = json.dumps(data)
    print(type(data))
    aas_df, aas_name, aas_id, aas_name = read_aas(
        data, aas, submodels, assets, conceptDescriptions, submodels_ids, metalabel
    )
    # aas_df_embeddings = encode(aas_df, model)
    aas_df = encode(aas_df, model)
    
    aas_df = predict_aas_type(aas_df, model, client_chroma_eclass)

    metadata, aas_index_str, se_content, se_embedding_name_definition = convert_to_list(
        aas_df
    )
    #print(metadata)
    collection = set_up_chroma(
        metadata,
        aas_index_str,
        se_content,
        se_embedding_name_definition,
        aas_name,
        client_chroma,
    )
    aas_response = post_aas_basyx(aas_file, data, aas_id, aas_name)

    return collection
