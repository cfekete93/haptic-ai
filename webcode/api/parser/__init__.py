import json

from flask import Blueprint, request
from flask_restx import Namespace, Resource, fields, marshal
from http import HTTPStatus

from service_classifier import service_classifier as sc

namespace = Namespace('Service Request Parser',
                      'Endpoints for parsing and processing user requests through AI classifiers',
                      path='/parser')

wild = fields.Wildcard(fields.String)
wildcard_fields = {'*': wild}

request_model = namespace.model(
    'Request',
    {
        # 'service': fields.Wildcard(fields.String, required=True, description='User Service Request')
        'text': fields.String(required=True, description='User Service Request')
    },
    strict=True
)

request_list_model = namespace.model(
    'RequestList',
    {
        'requests': fields.Nested(request_model, description='List of entities', as_list=True)
    },
    strict=True
)

api = Blueprint('api', __name__, url_prefix='/api-old')


@namespace.route('/service')
class ProcessRequest(Resource):
    """Respond to a user service request with a Service in JSON format"""

    @namespace.response(400, 'Bad User Request')
    @namespace.response(404, 'Entity not found')
    @namespace.response(500, 'Internal Server Error')
    @namespace.response(503, "Service Unavailable")
    # @namespace.marshal_with(request_model)
    @namespace.expect(request_model)
    def post(self):
        """Get a JSON formatted Service response for the given user service request"""
        # data = Service(Service.Type.INVALID, False, random_parm="value").to_dict()
        # return json.dumps(data), 200
        # return Service(Service.Type.INVALID, False, random_parm="value").to_dict()
        data = json.loads(request.data)
        user_request = data.get('text')
        service = sc.parse_request(user_request)
        return service.to_dict()

    @namespace.response(501, 'Not Implemented')
    @namespace.marshal_with(request_model)
    def get(self):
        namespace.abort(501)


@namespace.route('/services')
class ProcessRequests(Resource):
    """Respond to multiple user service requests via one API query with multiple JSON formatted objects"""

    @namespace.response(400, 'Bad User Request')
    @namespace.response(500, 'Internal Server Error')
    @namespace.response(503, "Service Unavailable")
    @namespace.marshal_list_with(request_list_model)
    def get(self):
        """Get an array of JSON formatted Service responses for the given user service requests"""
        response_list = []
        response_list.append(Service(Service.Type.INVALID, False, random_parm="value").to_json())
        response_list.append(Service(Service.Type.VIEW, True, coin="ETH").to_json())

        return {
            'entities': response_list,
            'total_records': len(response_list)
        }
