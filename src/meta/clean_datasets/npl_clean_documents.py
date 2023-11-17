def get_docs_as_lists_of_lines(lines):
    docs = []

    index = 0
    while index < len(lines):

        if lines[index][0].isdigit():
            current_doc = []
            index += 1

            while index < len(lines):
                current_doc.append(lines[index])
                index += 1

                if "/" in lines[index]:
                    docs.append(current_doc)
                    break

        index += 1

    return docs

def get_docs_as_strings(doc_lists):

    doc_strings = []

    for doc_list in doc_lists:
        doc_string = " ".join(doc_list)
        doc_string = doc_string.replace("\n", "")
        doc_strings.append(doc_string)

    return doc_strings


with open("../../../datasets/npl/doc-text") as f:
    lines = f.readlines()
    docs_list = get_docs_as_lists_of_lines(lines)
    docs_strings = get_docs_as_strings(docs_list)

with open("../../../datasets/npl/documents.txt", "a") as f:
    for doc_string in docs_strings:
        f.write(doc_string + "\n")
