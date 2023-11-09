import unittest
from text_authorship_classifier import main

class TestTextAuthorshipClassifier(unittest.TestCase):

    def test_authorship_classifier(self):
        # Provide test input text
        test_text = """
        This is a test text. It doesn't matter what it contains.
        """
        
        # Redirect stdout to capture the print statements
        import sys
        from io import StringIO
        original_stdout = sys.stdout
        sys.stdout = StringIO()

        # Run the classifier with the test input
        main(test_text)

        # Get the printed output
        output = sys.stdout.getvalue()
        
        # Assert that the expected output is in the result
        self.assertIn("The text was written by", output)

        # Restore original stdout
        sys.stdout = original_stdout

if __name__ == '__main__':
    unittest.main()