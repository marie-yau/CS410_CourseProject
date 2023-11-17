import re


def get_relevances_as_string(lines):
    relevances = []

    index = 1
    while index < len(lines):

        relevance = []

        while index < len(lines):
            relevance.append(lines[index])
            if "/" in lines[index]:
                relevances.append(relevance)
                index += 1
                break
            index += 1

        index += 1

    relevances_string = []
    for relevance in relevances:
        relevance = " ".join(relevance)
        relevance = relevance.replace("\n", " ")
        relevance = relevance.replace("/", " ")
        relevances_string.append(relevance)

    return relevances_string


def get_relevances_as_digits(relevances_string):
    relevances_list = []

    for relevance_string in relevances_string:
        relevance_string = re.sub(' +', ' ', relevance_string)
        relevance_string = relevance_string.strip()
        relevance_list = relevance_string.split(" ")
        relevances_list.append(relevance_list)

    return relevances_list


with open("../../../datasets/npl/rlv-ass") as f:
    lines = f.readlines()
    relevances_string = get_relevances_as_string(lines)
    relevances_list = get_relevances_as_digits(relevances_string)

with open("../../../datasets/npl/relevances.txt", "a") as f:
    for index, relevance_list in enumerate(relevances_list):
        for relevance in relevance_list:
            f.write(str(index + 1) + " " + relevance + "\n")

