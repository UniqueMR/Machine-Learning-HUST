def predRes(test, model):
    pred = model.predict(test.toarray())
    typeLen = len(pred[0])
    index = range(typeLen)
    res = []
    for item in pred:
        maxId = 0
        for id in index:
            if item[id] > item[maxId]:
                maxId = id
        res.append(maxId)
    fsave = open('./data/result.txt',mode='a')
    for item in res:
        fsave.write(str(item)+'\n')
    fsave.close()