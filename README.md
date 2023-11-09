# AI-Author-Prediction
## Text Authorship Classifier

This is a Python script that uses a simple text authorship classifier to determine the most likely author of a given text snippet. The classifier is implemented using the scikit-learn library and relies on the Multinomial Naive Bayes algorithm. It is trained on the works of three authors: Emma Goldman, Matthew Hensen, and Tingfang Wu.

## Prerequisites

Make sure you have Python installed on your system. You will also need the following libraries:

- [scikit-learn](https://scikit-learn.org/stable/): A machine learning library for Python.
- [modules.goldman_emma_raw](modules/goldman_emma_raw.py): A module containing text data from Emma Goldman.
- [modules.hensen_matthew_raw](modules/hensen_matthew_raw.py): A module containing text data from Matthew Hensen.
- [modules.wu_tingfang_raw](modules/wu_tingfang_raw.py): A module containing text data from Tingfang Wu.

You can install scikit-learn using pip:

```bash
pip install scikit-learn
```

## Usage

1. Ensure that you have the required Python libraries and data modules available in your environment.

2. Copy the provided code into a Python file (e.g., `text_authorship_classifier.py`).

3. Run the script:

```bash
python text_authorship_classifier.py
```

4. The script will output the predicted author based on the provided text. The example text in the script is:

```python
text_input = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freezing spray broke over the ship, encasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""
```

You can replace this text with your own input text to see which author the classifier predicts.

## How It Works

1. The script imports the necessary libraries and data modules.

2. It uses a Bag of Words (BoW) vectorizer to convert the text data from the authors into numerical features.

3. The script then trains a Multinomial Naive Bayes classifier on the vectorized text data from the three authors.

4. After training, it predicts the most likely author for the provided input text and outputs the result.

5. The script also displays the predicted probabilities for each author.

## Author Labels

- Emma Goldman (Label 1)
- Matthew Hensen (Label 2)
- Tingfang Wu (Label 3)

## Example Output

```
The text was written by Tingfang!
Emma: 4.1939939248959375e-05
Matthew: 5.682019518874032e-07
Tingfang: 0.9999563714903694
```

In the example output, the classifier predicts that the input text was most likely written by Tingfang Wu with a high probability. Emma Goldman and Matthew Hensen have very low probabilities.

Feel free to modify the code to use your own text data or expand the classifier with additional authors.
