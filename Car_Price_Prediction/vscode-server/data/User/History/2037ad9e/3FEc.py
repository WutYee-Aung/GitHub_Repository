from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route('/')
# def hello_world():
    # return "Hello World"

# if __name__ == '__main__':
    # app.run()

app = Flask(__name__)

@app.route('/',methods=['Get'])
def hello_world():
    # return "Hello World"
    # return render_template(index.html)
    return render_template('/templates/index.html')

if __name__ == '__main__':
    app.run()