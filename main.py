import flask
from flask import request, jsonify
from grammarFunc import grammerFunc
app = flask.Flask(__name__)


@app.route('/grammar', methods=['POST'])
def main():
    try:
        output = grammerFunc(request.form.get("sentence"))
        return jsonify({
            'status': 200,
            'body': {'value': output},
        })
    except Exception as e:
        return jsonify({
            "status": 500,
            "message": e
        })


@app.route("/", methods=["GET"])
def mainPage():
    return "<h1>Web Server is working!</h1>"


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
