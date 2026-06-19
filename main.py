"""main"""

# pylint: disable=import-error


from rsms import hello
from rsms import PotK


def main():
    """main"""
    print(hello())
    PotK("R1", [1.0])
    PotK("R6R12", [6.0, 12.0])
    PotK("R1R6R12", [1.0, 6.0, 12.0])


if __name__ == "__main__":
    main()
