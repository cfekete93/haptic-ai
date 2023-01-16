Haptics Dao AI Models
=====================

Note, these models present for Haptics DAO are currently in development and are subject to change and complete re-writes
without notice.

Requirements
------------

The following python packages should be installed to run/build this project. It's recommended to pip install these
inside of a python virtualenv if they're not already present in your system's managed packages.

```
# Basic requirements to run/build project

Flask==2.1.2
Werkzeug==2.1.2
tensorflow
```

Running
-------

# Building
To build the AI model run `./main.py build` from the root of the project. This may take some time so let it run as
it is very CPU intensive.

# Running Tests through CLI
To run some basic tests of the intent classifier run `./main.py test regular` and update the test_ic
function in main.py to test whatever statements you wish.

# Running REST API Tests through CLI
To run some basic tests of the intent classifier called from the REST API run `./main.py test api` and update the
test_sc function in main.py to test whatever statements you wish.

# Running Flask to Test REST API
To run a test instance of the REST API run `./demo.sh` and go to the subdomain `/api/doc` based on the address
the local WSGI server is running on. For instance, if it's running on localhost at port 5000 open the following URL,
`http://localhost:5000/api/doc` in a web browser.

Alternatively, run `./haptic_ai.py` and go to the same subdomain.
