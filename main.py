#!/usr/env/python
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from modules.goldman_emma_raw import goldman_docs
from modules.hensen_matthew_raw import hensen_docs
from modules.wu_tingfang_raw import wu_docs


def main():
    text_input = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""
    bow_vectorizer = CountVectorizer()
    test_text = goldman_docs() + hensen_docs() + wu_docs()
    # bow_vectorizer to train (fit) and vectorize (transform) all text
    input_text_vectors = bow_vectorizer.fit_transform(test_text)
    input_text_labels = [1] * 154 + [2] * 141 + [3] * 166

    # Vectorize input using transform() method
    author_probaility_vector = bow_vectorizer.transform([text_input])

    print(goldman_docs()[49])
    print(hensen_docs()[49])
    print(wu_docs()[49])

    # Implement a Naive Bayes classifier using MultinomialNB
    input_text_classifier = MultinomialNB()

    # Train input_text_classifier on input_text_vectors and input_text_labels using the classifier's fit() method
    input_text_labels = ["Emma"] * 154 + ["Matthew"] * 141 + ["Tingfang"] * 166
    input_text_classifier.fit(input_text_vectors, input_text_labels)

    # Prediction of input author:
    predictions = input_text_classifier.predict(author_probaility_vector)

    author = predictions[0] if predictions[0] else "someone else"

    author_probabilites_predictions = input_text_classifier.predict_proba(author_probaility_vector)
    print("The text was written by {}!".format(author))
    print(input_text_labels[0] + ": " + format(author_probabilites_predictions[0][0]))
    print(input_text_labels[1] + ": " + format(author_probabilites_predictions[0][1]))
    print(input_text_labels[2] + ": " + format(author_probabilites_predictions[0][2]))

if __name__ == "__main__":
    main()
