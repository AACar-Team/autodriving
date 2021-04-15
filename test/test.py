# yield的作用主要类似return，但是return之后不会保存任何状态，直接退出函数，yield会记录当前在哪里退出的，然后记录，下次调用的时候可以回到这个位置
def foo():
    print("starting...")
    while True:
        res = yield 4
        print(res)


# g = foo()
#
# print(next(g))
# print("*"*20)
# print(next(g))

def h():
    print('study yield')
    yield 5
    print('go on!')



c = h()
d1 = next(c)  # study yield
d2 = next(c)
