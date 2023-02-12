from src.lh_tool.Iterator import SingleProcess, MultiProcess


def process(a, b, opt='+'):
    if opt == '+':
        return a + b
    else:
        return a - b


def test_iterator():
    print('Test lh_tool.MultiProcess')
    a = [i for i in range(10)]
    b = [i for i in range(10)]
    result_list = SingleProcess(process).run(a, b, opt='+')
    print(result_list)

    result_list = MultiProcess(process, nprocs=1).run(a=a, b=b, opt='+')
    print(result_list)


if __name__ == '__main__':
    test_iterator()
