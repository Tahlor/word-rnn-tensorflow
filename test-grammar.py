import language_check


def correct_grammar(poem):
    tool = language_check.LanguageTool('en-US')
    matches = tool.check(poem)
    for i in matches:
        print(matches[i])
    new_poem = language_check.correct(poem, matches)
    return new_poem


poem = "This isflk th e poem, a efficient poem."

x = correct_grammar(poem)
print(x)
