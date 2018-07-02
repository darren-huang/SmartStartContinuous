import os


if __name__ == "__main__":
    print(os.path.abspath(__file__))
    from tests.testing.file import __file__
    print(os.path.abspath(__file__))