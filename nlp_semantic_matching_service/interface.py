import json
import pickle
from typing import List

from pydantic import BaseModel
import fastapi
from fastapi.responses import JSONResponse
import uvicorn
import basyx.aas.model
import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import semantic_matching_interface.interface
import semantic_matching_interface.query
import semantic_matching_interface.response

from nlp_pipeline import classify_elements, get_right_collection, classify_elements_one_aas
from compare_semantic_id import compare_semantic_ids, compare_semantic_ids_one_aas
from get_database import ask_database, delete_collection
from database_build import index_corpus
from database_idta import index_idta_submodels
from create_idta_submodels import create_idta_submodel

class CreateIdtaSubmodelRequest(BaseModel):
    submodel_name: str
    set_threshold: float

class NlpSemanticMatchingServiceInformation(BaseModel):
    matching_methods: List[semantic_matching_interface.response.SemanticMatchingServiceInformation   ]

# Create own class for the resonse of the semantic matching service which extends the class from response.py
class NlpSubmodelElementMatch(semantic_matching_interface.response.SubmodelElementMatch): 
    preferred_name: str
    definition: str
    semantic_id: str
    submodel_identifier_id: str

class SemanticMatchingService(
    semantic_matching_interface.interface.AbstractSemanticMatchingInterface
):
    # Get Chroma DB Client with indexed AASs
    

    #AWS


    client_chroma = chromadb.Client(
        Settings(
            chroma_api_impl="rest",
            # Chroma Server auf dem die Sentence Embeddings der VWS gespeichert werden sollen
            chroma_server_host="",
            chroma_server_http_port=8000,
        )
    )

    client_chroma_eclass = chromadb.Client(
        Settings(
            chroma_api_impl="rest",
            # Chrom Server auf dem die ECLASS Sentecne Embeddings gespeichert werden sollen
            chroma_server_host="",
            chroma_server_http_port=8000,
        )
    )

    client_chroma_idta_submodels = chromadb.Client(
        Settings(
            chroma_api_impl="rest",
            # Chrom Server auf dem die IDTA Embeddings der Teilmodelle gespeichert werden sollen
            chroma_server_host="",
            chroma_server_http_port=8000,
        )
    )
    

    def semantic_matching_service_information(self):
        matching_methods_list = [
            {"matching_method": "NLP without Metadata",
             "matching_algorithm": "Semantic search, k-nearest-neighbor with squared L2 distance (euclidean distance), with model gart-labor/eng-distilBERT-se-eclass",
             "required_parameters": ["semanticId", "preferredName", "definition"],
             "optional_parameters": ["unit", "datatype"]},
            {"matching_method": "NLP with Metadata",
             "matching_algorithm": "Semantic search, k-nearest-neighbor with squared L2 distance (euclidean distance), with model gart-labor/eng-distilBERT-se-eclass and classifier with metadata (unit and or datatype)",
             "required_parameters": ["semanticId", "preferredName", "definition"],
             "optional_parameters": ["unit", "datatype"]},
            {"matching_method": "Semantic equivalent, same semantic Id",
             "matching_algorithm": "None",
            "required_parameters": ["semanticId"],
            "optional_parameters": ["preferredName", "definition", "unit", "datatype"]}  
        ]      
        matching_methods = []
        
        for method in matching_methods_list:
            matching_methods.append(
                #semantic_matching_interface.response.SingleSemanticMatchingServiceInformation(
                semantic_matching_interface.response.SemanticMatchingServiceInformation(
                    matching_method=method["matching_method"],
                    matching_algorithm=method["matching_algorithm"],
                    required_parameters=method["required_parameters"],
                    optional_parameters=method["optional_parameters"]
            )
        )
        print(matching_methods)
        #return semantic_matching_interface.response.SemanticMatchingServiceInformation(
        return NlpSemanticMatchingServiceInformation(
            matching_methods=matching_methods,
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
            print(matching_result)
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
    
class AasPreparingHandling (semantic_matching_interface.interface.AbstractSemanticMatchingInterface):
    def __init__(self):

        self.router = fastapi.APIRouter()
        self.router.add_api_route(
            "/post_aas",
            self.post_aas,
            methods=["POST"]
        )
        self.router.add_api_route(
            "/get_aas",
            self.get_aas,
            methods=["GET"]
        )
        self.router.add_api_route(
            "/delete_aas",
            self.delete_aas,
            methods=["DELETE"]
        )

        self.router.add_api_route(
            "/post_idta_submodels",
            self.post_aas_idta_submodel,
            methods=["POST"]
        )

        self.router.add_api_route(
            "/create_idta_submodels",
            self.create_new_idta_submodel,
            methods=["POST"]
        )

        self.model = SentenceTransformer("gart-labor/eng-distilBERT-se-eclass")
        #self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        with open("nlp_semantic_matching_service/metadata.pickle", "rb") as handle:
            #global metalabel
            self.metalabel = pickle.load(handle)

        #global client_chroma
        

        #AWS



        self.client_chroma = chromadb.Client(
            Settings(
                chroma_api_impl="rest",
                # Chrom Server auf dem die Sentence Embeddings der VWS gespeichert werden sollen
                chroma_server_host="",
                chroma_server_http_port=8000,
            )
        )

        self.client_chroma_eclass = chromadb.Client(
            Settings(
                chroma_api_impl="rest",
                # Chrom Server auf dem die ECLASS Embeddings gespeichert werden sollen
                chroma_server_host="",
                chroma_server_http_port=8000,
            )
        )

        self.client_chroma_idta_submodels = chromadb.Client(
            Settings(
                chroma_api_impl="rest",
                # Chrom Server auf dem die IDTA Embeddings der Teilmodelle gespeichert werden sollen
                chroma_server_host="",
                chroma_server_http_port=8000,
            )
        )
        """
        self.client_chroma = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=".database/"
                # chroma_server_host muss angepasst werden nach jedem Neustart AWS
            )
        )

        self.client_chroma_idta_submodels = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=".database/"
            )
        )
        """

        

    async def post_aas(self, aas: fastapi.UploadFile = fastapi.File(...)):

        print(type(aas))
        data = json.load(aas.file)
        print(type(data))
        
        # Mit ECLASS collection = index_corpus(aas, data, self.model, self.metalabel, self.client_chroma, self.client_chroma_eclass)

        collection = index_corpus(aas, data, self.model, self.metalabel, self.client_chroma, self.client_chroma_eclass)
        ready = 'AAS ready'
        return ready
    
    async def get_aas(self):
        aas = ask_database(self.client_chroma)
        return aas

    async def delete_aas(self, aas_id: str):
        aas_in_database = delete_collection(self.client_chroma, aas_id)
        return aas_in_database
    
    # Create Chroma DB for IDTA Submodels
    async def post_aas_idta_submodel(self, aas: fastapi.UploadFile = fastapi.File(...)):

        print(type(aas))
        data = json.load(aas.file)
        print(type(data))

        # Mit ECLASS  collection = index_corpus(aas, data, self.model, self.metalabel, self.client_chroma_idta_submodels, self.client_chroma_eclass)
        collection = index_idta_submodels(aas, data, self.model, self.metalabel, self.client_chroma_idta_submodels)
        ready = 'AAS ready'
        return ready
    
    async def create_new_idta_submodel(
            self, 
            submodel_name: str = fastapi.Form(...),
            set_threshold: float = fastapi.Form(...),
            #request: CreateIdtaSubmodelRequest,
            aas: fastapi.UploadFile = fastapi.File(...),
        ):

        print(type(aas))
        data_dict = json.load(aas.file)
        print(type(data_dict))

        aas_with_idta, report = create_idta_submodel(
            data_dict, 
            self.model, 
            self.client_chroma_idta_submodels, 
            submodel_name, 
            set_threshold)
        
        aas_file_name = 'final_aas.json'

        headers = {
            "Content-Disposition": f"attachment; filename={aas_file_name}"
        }
        response = JSONResponse(content=aas_with_idta, headers=headers)

        report_name = 'matching_report.json'

        headers_report = {
            "Content-Disposition": f"attachment; filename={report_name}"
        }
        #response = [JSONResponse(content=aas_with_idta, headers=headers), JSONResponse(content=report, headers=headers_report)]

        return response


if __name__ == "__main__":
    APP = fastapi.FastAPI()
    APP.include_router(SemanticMatchingService().router)
    APP.include_router(AasPreparingHandling().router)
    uvicorn.run(APP, host="127.0.0.1", port=8002)
