import re
import matplotlib.pyplot as plt
import numpy as np


def main():
    file = open('screenlog.0','r')
    list = []
    n = 0
    for line in file:
        m = re.search('Loss',line)
        n += 1
        if m and n%10 == 0:
            list.append(re.findall("[0-9]+\.[0-9]+", line)[5])
            print(re.findall("[0-9]+\.[0-9]+", line)[5])
    file.close()
    plt.plot(list)
    plt.show()

if __name__ == '__main__':
    main()
