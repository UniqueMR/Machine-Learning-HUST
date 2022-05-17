#定义欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))