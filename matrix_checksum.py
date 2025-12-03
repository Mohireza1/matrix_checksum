import numpy as np
import json
import argparse

# A = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
# B = np.arange(7, 13, dtype=np.float32).reshape(2, 3)
# C = A @ B


class ChecksumMatrix:
    def __init__(self, A, B, C):
        self.C = C
        self.c = A.sum(axis=0)  # sum over columns of A → shape (3,)
        self.r = B.sum(axis=1)  # sum over rows of B → shape (3,)

        self.r_c = A @ self.r
        self.c_c = self.c @ B

        col_check = (self.c_c == C.sum(axis=0)).astype(np.float32)
        row_check = (self.r_c == C.sum(axis=1)).astype(np.float32)

        col_errors = np.where(col_check == 0)[0]
        row_errors = np.where(row_check == 0)[0]
        print("\n")
        self.col_idx = col_errors[0] if len(col_errors) > 0 else None
        self.row_idx = row_errors[0] if len(row_errors) > 0 else None

        print("Original C:\n", C)
        print("\n")
        print("sum of C rows: ", C.sum(axis=1))
        print("Predicted sum of C rows: ", self.r_c)
        print("sum of C cols: ", C.sum(axis=0))
        print("Predicted sum of C columns: ", self.c_c)
        print("\n")

    def verify(self):
        checker = self.r @ self.c
        if self.row_idx is None and self.col_idx is None:
            print("The output is correct")
            return

        print(
            f"The invalid output lies at row {self.row_idx} and column {self.col_idx}"
        )

    def fix(self):
        if self.row_idx is None or self.col_idx is None:
            print("No correction needed; checksum passed.")
            return

        self.C[self.row_idx, self.col_idx] += (
            self.r_c[self.row_idx] - self.C.sum(axis=1)[self.row_idx]
        )
        print("Corrected C:\n", self.C)


def main():
    parser = argparse.ArgumentParser(
        description="A simple matrix checksum verification and fix.",
        epilog="Example: python matrix_checksum.py verify matrices.json",
    )

    parser.add_argument(
        "action",
        choices=["verify", "fix"],
        help="The type of operation you want to run on the output. can be 'verify' or 'fix'.",
    )

    parser.add_argument(
        "filename", help="Name of the JSON file containing the matrices."
    )

    args = parser.parse_args()

    with open(f"{args.filename}") as f:
        data = json.load(f)

    A = np.array(data["A"], dtype=np.float32)
    B = np.array(data["B"], dtype=np.float32)
    C = np.array(data["C"], dtype=np.float32)

    matrix = ChecksumMatrix(A, B, C)

    if args.action == "verify":
        matrix.verify()
    else:
        matrix.fix()


if __name__ == "__main__":
    main()
