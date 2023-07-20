import datetime

def select_idta_submodels(submodel_name):

    if submodel_name == 'https://admin-shell.io/zvei/nameplate/2/0/Nameplate':
        
        timestamp = datetime.datetime.now().timestamp()
        timestamp = str(int(timestamp))
        submodel_id = 'idta_digital_nameplate_' + timestamp
        aas_submodel_information = {
          "keys": [
            {
              "type": "Submodel",
              "local": "true",
              "value": submodel_id,
              "index": 0,
              "idType": "IRI"
            }
          ]
        }
        """
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
                "id": id
            },
            "idShort": "Nameplate_IDTA",
            "modelType": {
                "name": "Submodel"
            },
            "kind": "Instance",
            "submodelElements": []
        }
        """

        return aas_submodel_information, submodel_id