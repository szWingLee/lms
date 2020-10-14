class Base:
    def __init__(self):
        print('base in')
        print('base out')


class A(Base):
    def __init__(self):
        print('a in')
        Base.__init__(self)
        print('a out')


class B(Base):
    def __init__(self):
        print('b in')
        super().__init__()
        print('b out')


class B1(Base):
    def __init__(self):
        print('b1 in')
        super().__init__()
        print('b1 out')


class C(B, B1, A):
    def __init__(self):
        print('c in')
        super(C, self).__init__()
        print('c out')


class D(A, B, B1):
    def __init__(self):
        print('d in')
        super().__init__()
        print('d out')


class E(B, A, B1):
    def __init__(self):
        print('e in')
        super().__init__()
        print('e out')


if __name__ == '__main__':
    C()
    print('')
    D()
    print('')
    E()
