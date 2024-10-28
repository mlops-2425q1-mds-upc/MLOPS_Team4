from Sentiment_Analysis.data_preprocessing import clean_text


def test_clean_text():
    input_text = "This is a sample #text with https://example.com URL, numbers 123, and @mentions!"
    expected_output = "sampl text url number"

    cleaned_text = clean_text(input_text)

    assert (
        cleaned_text == expected_output
    ), f"Expected '{expected_output}', but got '{cleaned_text}'"
