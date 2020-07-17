def _infer_type(ser, type_list=None):
    if not type_list:
        type_list = ["float64", "M8[us]"]
    for _type in type_list:
        try:
            _ = ser.astype(_type)
            return _type
        except (TypeError, ValueError):
            pass
    return "str"


def infer_data_types(df, type_list=None):
    type_dct = {}
    for k in df.columns:
        type_dct[k] = _infer_type(df[k], type_list=type_list)
    return type_dct


class FixDTypes:
    def __init__(self, type_lst=None, row_sample_size=30, random_state=0):
        self.type_lst = type_lst
        self.dtypes = None
        self.row_sample_size = row_sample_size
        self.random_state = random_state

    def get_sample_df(self, X, random_state=0):
        self.sample_df = X.sample(frac=1.0, random_state=self.random_state).head(
            self.row_sample_size
        )
        if hasattr(self.sample_df, "compute"):
            self.sample_df = self.sample_df.compute()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.get_sample_df(X)
        self.dtypes = infer_data_types(self.sample_df, type_list=self.type_lst)
        X = X.astype(self.dtypes)
        return X
