from flask import Blueprint
from flask_restx import Api
from webcode.api.parser import namespace as parser_ns

api = Blueprint('api', __name__, url_prefix='/api')

api_extension = Api(
    api,
    title='Haptics DAO AI - Developer API',
    version='0.0',
    description='Application for Development tutorial of the Haptics DAO AI REST API',
    doc='/doc'
)

api_extension.add_namespace(parser_ns)
