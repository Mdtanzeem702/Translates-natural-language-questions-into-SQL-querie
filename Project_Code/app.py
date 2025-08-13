from flask import Flask, render_template, request, jsonify
from modules.llm_handler import my_llm_response


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    
    # Placeholder response from handler (you'll add LLM later)
    obj=my_llm_response()
    result = obj.handle_question(question=user_question)
    print("result in app.py: ",result)


    return jsonify({"answer": result["answer"], "query": [result["query"]]})

if __name__ == '__main__':
    app.run(debug=True)
