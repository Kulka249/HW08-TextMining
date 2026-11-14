def get_formatted_text(filename):
    lines = []

    with open(filename, "r") as f:
        lines = f.readlines()
        lines = ["__" + line.strip().lower() + "__" for line in lines]

    return lines



def get_ngrams(line, n):
    ngrams = []  # init a list

    N = len(line)
    L = n
    for k in range(N - L + 1):
        ngrams.append(line[k : k + L])

    return ngrams

def get_dict(filename, n):
    ngram_dict = {}
    for line in get_formatted_text(filename):
        ngrams = get_ngrams(line, n)
        for gram in ngrams:
            ngram_dict[gram] = ngram_dict.get(gram, 0) + 1
    return ngram_dict

def top_N_common(filename, N, n, threshold=0):
    return dict(sorted({key: value for key, value in get_dict(filename, n).items() if value >= threshold}.items(), key=lambda item: (item[1], item[0]), reverse=True)[:N])

def get_all_dicts(filename_list, n):
    lang_dicts = []

    for lang in filename_list:
        langDict = get_dict(lang, n)
        lang_dicts.append(langDict)

    return lang_dicts

def dict_union(listOfDicts):
    union_ngrams = set()
    for dictionary in listOfDicts:
        union_ngrams.update(dictionary.keys())
    return sorted(union_ngrams)


def get_all_ngrams(langFiles, n):
    all_ngrams = []
    all_ngrams = dict_union(get_all_dicts(langFiles, n))

    return all_ngrams

def compare_langs(test_file, langFiles, N, n=3):
    lang_match = ''
    max_matches = -1

    for lang in langFiles:
        lang_grams = set(top_N_common(lang, N, n))
        matches = len(set(top_N_common(test_file, N, n)).intersection(lang_grams))
        if matches > max_matches:
            max_matches = matches
            lang_match = lang

    return lang_match


if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join, splitext

    # Testing top_20_common
    path = join("ngrams", "english.txt")
    print("Top 10 common n-grams in english.txt:")
    print(top_N_common(path,20,4))
    print("\n")
    print("Top 10 common n-grams in english.txt with threshold 50:")
    print(top_N_common(path,20,4,threshold=50))
    print("\n")
    path = "ngrams"
    file_list = [f for f in listdir(path) if isfile(join(path, f))]
    file_list.remove("mystery.txt")
    path_list = [join(path, f) for f in file_list]
    print(get_all_ngrams(path_list, 4)[:50])  # list of all 4-grams spanning all languages
    test_file = join(path, "mystery.txt")
    print(compare_langs(test_file, path_list, 20))  # determine language of mystery file
