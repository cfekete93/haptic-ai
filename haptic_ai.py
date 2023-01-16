#!/usr/bin/env python3

from flask import Flask

# from webcode.apps.parser import parser
from webcode.api import api

app = Flask(__name__)
app.config['RESTPLUS_MASK_SWAGGER'] = False

app.register_blueprint(api)

# Run haptic-ai application
if __name__ == '__main__':
    app.run()
