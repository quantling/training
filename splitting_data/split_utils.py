class NotEnoughDiskSpaceError(Exception):
    """Custom exception for insufficient disk space."""
    def __init__(self, message="There is likely not enough disk space left. Dumped the current state of the counters."):
        self.message = message
        super().__init__(self.message)


def replace_special_chars(word):
    """This function is used to replace special characters in a word. It is in this file so that characters for both multiprocessing and single processing are the same"""

    # I just copied this from the create_vtl corpus but importing was too much work
    cleaned_word = (
        word.replace(".", "")
        .replace(",", "")
        .replace("?", "")
        .replace("!", "")
        .replace(":", "")
        .replace(";", "")
        .replace("(", "")
        .replace(")", "")
        .replace('"', "")
        .replace("'", "")
        .replace('"', "")
        .replace("„", "")
        .replace("“", "")
        .replace("”", "")
        .replace("‘", "")
        .replace("´", "")
        .replace("…", "")
        .replace("«", "")
        .replace("»", "")
        .replace("'", "")
        .replace("’", "")
    )
    return cleaned_word