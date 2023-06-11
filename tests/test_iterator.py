import asyncio
from src.lh_tool.Iterator import SingleProcess, MultiProcess, AsyncProcess, AsyncMultiProcess, ParallelProcess


def process(a, b, opt='+'):
    if opt == '+':
        return a + b
    else:
        return a - b


def process_batch(a_list, b_list, opt='+'):
    res_list = []
    for a, b in zip(a_list, b_list):
        if opt == '+':
            res = a + b
        else:
            res = a - b
        res_list.append(res)
    return res_list


async def async_process(a, b, opt='+'):
    await asyncio.sleep(1)
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

    result_list = MultiProcess(process, nprocs=10).run(a=a, b=b, opt='+')
    print(result_list)

    result_list = AsyncProcess(async_process).run(a=a, b=b, opt='+')
    print(result_list)

    result_list = AsyncMultiProcess(async_process, nprocs=10).run(a=a, b=b, opt='+')
    print(result_list)

    ret_list = ParallelProcess(process_batch, nprocs=5).run(a_list=a, b_list=b, opt='+')
    result_list = []
    for ret in ret_list:
        result_list.extend(ret)
    print(result_list)


if __name__ == '__main__':
    test_iterator()
