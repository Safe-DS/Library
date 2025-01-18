def _get_similar_strings(string: str, valid_strings: list[str]) -> list[str]:
    from difflib import get_close_matches

    close_matches = get_close_matches(string, valid_strings, n=3)

    if close_matches and close_matches[0] == string:
        return close_matches[0:1]
    else:
        return close_matches
