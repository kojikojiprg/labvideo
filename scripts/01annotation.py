import json

info_filename = "./info.json"
with open(info_filename) as fp:
    info = json.load(fp)

print(info)
fp = open("annotation.tsv")
head = next(fp)
a_filename = "annotation.json"
annotation = {}
vid = None
print(head.rstrip().split("\t"))
for line in fp:
    arr = line.rstrip().split("\t")
    if arr[1] != "":
        # print(arr)
        vid = arr[1]
        if arr[1] in info:
            print("hit ", arr[2], info[arr[1]])
        else:
            print("miss", arr[2])
        annotation[vid] = []
    if len(arr) > 3:
        aid = arr[3]
        atype = arr[4]
        text = arr[5]
        timestamp = arr[6]
        if len(arr) > 7:
            reply = arr[7]
        else:
            reply = None
        if len(arr) > 8:
            url = arr[8]
        else:
            url = None

        print(text)
        annotation[vid].append(
            {
                "aid": aid,
                "type": atype,
                "text": text,
                "time": timestamp,
                "url": url,
            }
        )

with open(a_filename, mode="w") as fp:
    json.dump(annotation, fp)
