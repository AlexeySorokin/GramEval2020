def process_feats(s):
    if s == "_":
        return dict()
    return dict([x.split("=") for x in s.split("|")])


def read_UD_infile(infile):
    answer, curr_sent = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                if len(curr_sent) > 0:
                    answer.append(curr_sent)
                curr_sent = []
                continue
            elif line[0] == "#":
                continue
            else:
                data = line.split("\t")
                if not data[0].isdigit():
                    continue
                data[5] = process_feats(data[5])
                curr_sent.append(data)
    if len(curr_sent) > 0:
        answer.append(curr_sent) 
    return answer