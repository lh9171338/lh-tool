from lh_tool.MultiProcessing import MultiProcessing

def process(args):
    a, b, opt = args
    if opt == '+':
        return a + b
    else:
        return a - b


def test_multiprocessing():
    print('Test lh_tool.MultiProcessing')
    a = [i for i in range(10)]
    b = [i for i in range(10)]
    result_list = MultiProcessing(process, 10).run(a, b, '+')
    print(result_list)


if __name__ == '__main__':
    test_multiprocessing()
