import json

import fastapi
import uvicorn
import basyx.aas.model
import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
import semantic_matching_interface.interface
import semantic_matching_interface.query
import semantic_matching_interface.response

from nlp_pipeline import classify_elements, get_right_collection, classify_elements_one_aas
from compare_semantic_id import compare_semantic_ids, compare_semantic_ids_one_aas

# Create own class for the resonse of the semantic matching service which extends the class from response.py
class NlpSubmodelElementMatch(semantic_matching_interface.response.SubmodelElementMatch): 
    preferred_name: str
    definition: str
    semantic_id: str

class SemanticMatchingService(
    semantic_matching_interface.interface.AbstractSemanticMatchingInterface
):
    # Get Chroma DB Client with indexed AASs
    client_chroma = chromadb.Client(
        Settings(
            chroma_api_impl="rest",
            # chroma_server_host muss angepasst werden nach jedem Neustart AWS
            chroma_server_host="3.75.91.79",
            chroma_server_http_port=8000,
        )
    )

    def semantic_matching_service_information(self):
        matching_methods_list = [
            {"matching_method": "NLP without Metadata",
             "matching_algorithm": "Semantic search, k-nearest-neighbor with squared L2 distance (euclidean distance), with model gart-labor/eng-distilBERT-se-eclass"},
            {"matching_method": "NLP with Metadata",
             "matching_algorithm": "Semantic search, k-nearest-neighbor with squared L2 distance (euclidean distance), with model gart-labor/eng-distilBERT-se-eclass and classifier with metadata (unit and or datatype)"},
            {"matching_method": "Semantic equivalent, same semantic Id", "matching_algorithm": "None"}
        ]
        matching_methods = []
        """
        for method in matching_methods_list:
            matching_methods.append(
                #semantic_matching_interface.response.SingleSemanticMatchingServiceInformation(
                semantic_matching_interface.response.SingleSemanticMatchingServiceInformation(
                    matching_method=method["matching_method"],
                    matching_algorithm=method["matching_algorithm"]
                )
            )
        """
        return semantic_matching_interface.response.SemanticMatchingServiceInformation(
            #matching_methods=matching_methods,
            matching_method='NLP without Metadata',
            matching_algorithm='Semantic search, k-nearest-neighbor with squared L2 distance (euclidean distance), with model gart-labor/eng-distilBERT-se-eclass',
            required_parameters=["semanticId", "preferredName", "definition"],
            optional_parameters=["unit", "dataType"]
        )

    def semantic_query_asset_administration_shell(
            self,
            query: semantic_matching_interface.query.AssetAdministrationShellQuery,
            response: fastapi.Response
    ):
        collections = self.client_chroma.list_collections()
        queries = query.query_parameters
        semantic_id = None
        for element in queries:
            if element.attribute_name == 'preferredName':
                preferred_name = element.attribute_value
            elif element.attribute_name == 'definition':
                definition = element.attribute_value
            elif element.attribute_name == 'dataType':
                datatype = element.attribute_value
            elif element.attribute_name == 'unit':
                unit = element.attribute_value
            elif element.attribute_name == 'semanticId':
                semantic_id = element.attribute_value
        if 'unit' in locals():
            print('unit exists')
        else:
            unit = 'NotInQuery'
        if 'datatype' in locals():
            print('datatype exists')
        else:
            datatype = 'NotInQuery'
        return_matches = query.return_matches
        # search for same semantic id
        if semantic_id is None:
            response.status_code = fastapi.status.HTTP_400_BAD_REQUEST
            return
        results_semantic_equivalent = compare_semantic_ids(self.client_chroma, semantic_id)
        # do nlp
        results_with_nlp = classify_elements(collections, self.client_chroma, preferred_name, definition, unit,
                                             datatype, semantic_id, number_elements=1)

        # see if there are aas with same semantic id then requested, otherwise use nlp result
        results = []
        for result_homogen, result_heterogen in zip(results_semantic_equivalent, results_with_nlp):
            if result_homogen != 0:
                results.append(
                    result_homogen
                )
            else:
                results.append(
                    result_heterogen
                )
                # sort best matching aas based on matching distance
        sorted_results = sorted(results, key=lambda aas: aas['matching_distance'])
        # get best results based on number of requested aas
        best_results = sorted_results[0:return_matches]

        matching_result = []
        for result in best_results:
            matching_result.append(
                semantic_matching_interface.response.AssetAdministrationShellMatch(
                    matching_method=result['matching_method'],
                    matching_algorithm=result['matching_algorithm'],
                    matching_score=result['matching_distance'],
                    aas_identifier_id=result['aas_id'],
                    aas_identifier_id_type="IRI"
                )
            )
        return semantic_matching_interface.response.AssetAdministrationShellMatchingResponse(
            matching_result=matching_result
        )

    def semantic_query_submodel_element(
            self,
            query: semantic_matching_interface.query.SubmodelElementQuery
    ):
        collections = self.client_chroma.list_collections()
        queries = query.query_parameters
        for element in queries:
            if element.attribute_name == 'preferredName':
                preferred_name = element.attribute_value
            elif element.attribute_name == 'definition':
                definition = element.attribute_value
            elif element.attribute_name == 'datatype':
                datatype = element.attribute_value
            elif element.attribute_name == 'unit':
                unit = element.attribute_value
            elif element.attribute_name == 'semanticId':
                semantic_id = element.attribute_value
        if 'unit' in locals():
            print('unit exists')
        else:
            unit = 'NotInQuery'
        if 'datatype' in locals():
            print('datatype exists')
        else:
            datatype = 'NotInQuery'
        aas_id = query.aas_identifier_id

        number_elements = query.return_matches
        # get the right collection with the correct requested aas
        right_collection = get_right_collection(collections, aas_id)
        # Response if asked aas is not in collections
        if right_collection == ['NotFound']:
            matching_result = [
                semantic_matching_interface.response.SubmodelElementMatch(
                    matching_method="Not in database",
                    matching_algorithm="Not found",
                    submodel_identifier_id='Not found',
                    matching_score=0,
                    id_short_path=''
                )
            ]
            return semantic_matching_interface.response.SubmodelElementMatchingResponse(
                matching_result=matching_result,
                aas_identifier_id=query.aas_identifier_id,
                aas_identifier_id_type=query.aas_identifier_id_type
            )
        else:
            # get the collection with data from the right aas
            collection = self.client_chroma.get_collection(right_collection[0].name)
            # look if semantic equivalent semanctic_id is there
            result_semantic_equivalent = compare_semantic_ids_one_aas(collection, self.client_chroma, preferred_name,
                                                                      definition, unit, datatype, semantic_id)
            # than do nlp
            results_with_nlp = classify_elements_one_aas(collection, self.client_chroma, preferred_name, definition,
                                                         unit, datatype, semantic_id, number_elements)

        matching_result = []
        # same semantic_id, use this result
        if result_semantic_equivalent != 0:
            matching_result.append(
                NlpSubmodelElementMatch(
                    matching_method=result_semantic_equivalent['matching_method'],
                    matching_algorithm=result_semantic_equivalent['matching_algorithm'],
                    matching_score=result_semantic_equivalent['matching_distance'],
                    submodel_identifier_id=result_semantic_equivalent['submodel_id'],
                    id_short_path=result_semantic_equivalent["id_short_path"],
                    preferred_name = result_semantic_equivalent["preferred_name"],
                    definition = result_semantic_equivalent["definition"],  
                    semantic_id = result_semantic_equivalent["semantic_id"]
                )
            )
            return semantic_matching_interface.response.SubmodelElementMatchingResponse(
                matching_result=matching_result,
                aas_identifier_id=query.aas_identifier_id,
                aas_identifier_id_type=query.aas_identifier_id_type
            )
        # otherwise use nlp results
        else:
            matching_result = []
            for result in results_with_nlp:
                matching_result.append(
                    #semantic_matching_interface.response.SubmodelElementMatch(
                    NlpSubmodelElementMatch(
                        matching_method=result['matching_method'],
                        matching_algorithm=result['matching_algorithm'],
                        matching_score=result['matching_distance'],
                        submodel_identifier_id=result['submodel_id'],
                        id_short_path=result["id_short_path"],
                        preferred_name = result["preferred_name"],
                        definition = result["definition"],  
                        semantic_id = result["semantic_id"]
                    )
                )
            return semantic_matching_interface.response.SubmodelElementMatchingResponse(
                matching_result=matching_result,
                aas_identifier_id=query.aas_identifier_id,
                aas_identifier_id_type=query.aas_identifier_id_type
            )

    def semantic_match_objects(
            self,
            query: semantic_matching_interface.query.MatchObjectsQuery
    ):
        raise NotImplementedError


if __name__ == "__main__":
    APP = fastapi.FastAPI()
    APP.include_router(SemanticMatchingService().router)
    uvicorn.run(APP, host="127.0.0.1", port=8002)
