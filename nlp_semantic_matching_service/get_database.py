import requests
import pandas as pd

def get_values(submodel_element):
    # Auslesen der Submodel Element Werte
    #print(submodel_element)
    se_type = submodel_element["modelType"]["name"]
    se_semantic_id = submodel_element["semanticId"]["keys"][0]["value"]
    se_semantic_id_local = submodel_element["semanticId"]["keys"][0]["local"]
    se_id_short = submodel_element["idShort"]
    #id_short_path = id_short_path + '.' + se_id_short
    value = []
    se_value = submodel_element["value"]
    value.append(se_value)

    return se_type, se_semantic_id, se_semantic_id_local, se_id_short, value

def get_values_sec(
    sec_elements,
    this_submodel,
    content,
    aas_id,
    aas_name,
    submodel_id,
    submodel_id_short,
):

    collection_values = content[0]["value"]
    for element in collection_values:
        #nested_sec = []
        content = []
        content.append(element)

        se_type, se_semantic_id, se_semantic_id_local, se_id_short, value = get_values(
            collection_values[element]
        )

        if se_type == "SubmodelElementCollection":
            nested_sec = []
            submodel_element_dict = {
                "AASId": aas_id,
                "AASIdShort": aas_name,
                "SubmodelId": submodel_id,
                "SubmodelName": submodel_id_short,
                "SESemanticId": se_semantic_id,
                "SEModelType": se_type,
                "SEIdShort": se_id_short,
                "SEValue": value
                #"SEValue": nested_sec
            }
            nested_sec.append(submodel_element_dict)
            sec_elements.append(nested_sec)
            #nested_sec = []                        
            #nested_sec.append(submodel_element_dict)
            #sec_elements.append(submodel_element_dict)

            content = []
            content.append(collection_values[element])
            #print(content)
            # Rekursive Funktion -> so oft durchlaufen bis unterste Ebene der Collections erreicht ist, so werden verschachteltet SECs bis zum Ende ausgelesen

            this_submodel = get_values_sec(
                nested_sec,
                this_submodel,
                content,
                aas_id,
                aas_name,
                submodel_id,
                submodel_id_short
            )
            #sec_elements.append(nested_sec)

        else: 
            submodel_element_dict = {
                "AASId": aas_id,
                "AASIdShort": aas_name,
                "SubmodelId": submodel_id,
                "SubmodelName": submodel_id_short,
                "SESemanticId": se_semantic_id,
                "SEModelType": se_type,
                "SEIdShort": se_id_short,
                "SEValue": value
            }

            #nested_sec.append(submodel_element_dict)
            sec_elements.append(submodel_element_dict)

    this_submodel[-1]['SEValue'] = sec_elements

    #return df
    return this_submodel

def ask_database(client):
    # ask basyx server
    """
    aas_server = "http://3.83.126.51:4001/aasServer/shells"
    response = requests.get(aas_server)
    all_aas = response.json()
    print(type(all_aas))
    #print(all_aas)

    read_aas = []
    for aas in all_aas:
        this_aas = []
        aas_id = aas['identification']['id']
        aas_name = aas['idShort']
        new_aas_id = aas_id.replace('/', '%2F')
        url_submodels = aas_server + '/' + new_aas_id + '/aas/submodels'
        response = requests.get(url_submodels)

        all_submodels = response.json()
        for submodel in all_submodels:


            # Wie bekomme ich es hin das klar ist das ein submodel element zu einer submodel element collection gehört?
            
            this_submodel = []
            submodel_id_short = submodel['idShort']
            submodel_id = submodel['identification']['id']
            print(submodel_id_short)
            se_urls = url_submodels + '/' + submodel_id_short + '/submodel/submodelElements'
            #se_urls = url_submodels + '/' + submodel_id_short + '/submodel/values'
            response_submodel_elements = requests.get(se_urls)
            submodel_elements = response_submodel_elements.json()
            print(len(submodel_elements))

            for submodel_element in submodel_elements:
                content = []
                content.append(submodel_element)
                #print(content)
                (
                    se_type,
                    se_semantic_id,
                    se_semantic_id_local,
                    se_id_short,
                    value
                ) = get_values(submodel_element)

                submodel_element_dict = {
                    "AASId": aas_id,
                    "AASIdShort": aas_name,
                    "SubmodelId": submodel_id,
                    "SubmodelName": submodel_id_short,
                    "SESemanticId": se_semantic_id,
                    "SEModelType": se_type,
                    "SEIdShort": se_id_short,
                    "SEValue": value,
                }

                this_submodel.append(submodel_element_dict)
                #df = pd.concat([df, new_row], ignore_index=True)
                #print(df)

                # Wenn Submodel Element Collection dann diese Werte auch auslesen
                if se_type == "SubmodelElementCollection":
                    sec_elements = []
                    #df = get_values_sec(
                    this_submodel = get_values_sec(
                        #data,
                        #df_cd,
                        sec_elements,
                        this_submodel,
                        content,
                        aas_id,
                        aas_name,
                        submodel_id,
                        submodel_id_short,
                        #submodel_semantic_id
                    )
            #print(df)

            #submodel_dict = df.to_dict(orient='index')
            #this_aas.append(submodel_dict)
            this_aas.append(this_submodel)
            print(len(this_aas))

        read_aas.append(this_aas)
        print(len(read_aas))
                
       
    #all_aas_list = list(all_aas)
    #print(all_aas_list)
    """
    # read chroma
    
    collection_list = client.list_collections()
    aas_list = []
    for collection in collection_list:
        collection_name = collection.name
        collection = client.get_collection(collection_name)
        items = collection.count()
        first = collection.get(ids=['1'])
        aas_id = first['metadatas'][0]['AASId']
        aas_type = first['metadatas'][0]['AASType']
        print(aas_id)
        aas_content = first['metadatas'][0]['AASContent']
        #print(aas_content)
        aas_id_short = first['metadatas'][0]['AASIdShort']
        aas_dict = {'aas_id': aas_id, 'aas_id_short': aas_id_short, 'number_items': items, 'aas_content': aas_content, 'aas_type': aas_type}
        aas_list.append(aas_dict)
    
    return aas_list
    #return read_aas

def delete_collection(client, aas_id):
    #delete aas in basyx server
    aas_server = "http://3.83.126.51:4001/aasServer/shells"
    new_aas_id = aas_id.replace('/', '%2F')
    print(new_aas_id)
    delete_aas = aas_server + '/' + new_aas_id
    deleted = requests.delete(delete_aas)

    #delete chroma aas
    collections = client.list_collections()
    right_collection = []
    for collection in collections:
        try_collection = collection.get(where={'AASId': aas_id})
        try:
            collection_aas_id = try_collection['metadatas'][0]['AASId']
            right_collection.append(collection)
        except:
            print('Nix')
    for aas in right_collection:
        client.delete_collection(name=aas.name)
    collection_list = client.list_collections()
    aas_list = []
    for collection in collection_list:
        collection_name = collection.name
        collection = client.get_collection(collection_name)
        items = collection.count()
        first = collection.get(ids=['1'])
        aas_id = first['metadatas'][0]['AASId']
        aas_id_short = first['metadatas'][0]['AASIdShort']
        aas_dict = {'aas_id': aas_id, 'aas_id_short': aas_id_short, 'number_items': items}
        aas_list.append(aas_dict)

    return aas_list
