"""
Unit tests for data preprocessing functions in the Sentiment_Analysis package.
"""
# pylint: disable=import-error
from Sentiment_Analysis.data_preprocessing import clean_text


def test_clean_text():
    """
    Test the clean_text function for correct text preprocessing.
    """
    input_text = "This is a sample #text with https://example.com URL, numbers 123, and @mentions!"
    expected_output = "sampl text url number"

    cleaned_text = clean_text(input_text)

    assert (
        cleaned_text == expected_output
    ), f"Expected '{expected_output}', but got '{cleaned_text}'"
