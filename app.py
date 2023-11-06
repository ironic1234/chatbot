from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, jsonify

# Assuming you have a list of questions and their corresponding labels

labels = []

questions = []

# Open the file in read mode
with open("bioquestions.txt", "r") as file:
    # Read each line from the file and append it to the list
    for line in file:
        questions.append(line.strip())  # Use strip() to remove leading/trailing whitespace
        labels.append("Biology")

# Open the file in read mode
with open("geoquestions.txt", "r") as file:
    # Read each line from the file and append it to the list
    for line in file:
        questions.append(line.strip())  # Use strip() to remove leading/trailing whitespace
        labels.append("Geography")

# Open the file in read mode
with open("chemquestions.txt", "r") as file:
    # Read each line from the file and append it to the list
    for line in file:
        questions.append(line.strip())  # Use strip() to remove leading/trailing whitespace
        labels.append("Chemistry")

# Step 2: Data Preprocessing
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(questions)

# Step 3: Model Selection
model = MultinomialNB()

# Step 4: Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

def run(question):
    new_question = [question]
    new_question_vector = tfidf_vectorizer.transform(new_question)
    predicted_type = model.predict(new_question_vector)
    return predicted_type

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def attribute():
    data = request.get_json()  # Parse JSON data from the request
    user_input = data.get('text', '')

    if user_input:
        returntext = run(user_input)[0]
        return jsonify({'type': returntext})
    else:
        return jsonify({'error': 'Missing "text" parameter in the request'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0')

