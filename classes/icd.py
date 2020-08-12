import pandas as pd

def convert_to_icd9(dx_str):
    if dx_str.startswith("E"):
        # add decimal if 5 char or longer (inclusive of 'E')
        if len(dx_str) > 4:
            return dx_str[:4] + "." + dx_str[4:]
        else:
            return dx_str
    else:  # not E codes
        # add decimal if 4 digits or longer
        if len(dx_str) > 3:
            return dx_str[:3] + "." + dx_str[3:]
        else:
            return dx_str


def convert_to_3digit_icd9(dx_str):
    if dx_str.startswith("E"):
        if len(dx_str) > 4:
            return dx_str[:4]
        else:
            return dx_str
    else:  # not E codes
        if len(dx_str) > 3:
            return dx_str[:3]
        else:
            return dx_str


class ICD9Desc(dict):
    def __init__(self, file='data/CMS32_DESC_LONG_DX.txt', widths=[5,10000]):
        self.file = file
        self.widths = widths

    def get_dict(self):
        df = pd.read_fwf(self.file, widths=self.widths,
                         names=['ICD9CM','description'], index=False)
#        df['description'] = df.description.str.rstrip()
        df.set_index(df.ICD9CM.apply(convert_to_icd9), inplace=True)
        icd9cm_dict = df.description.to_dict()
        return icd9cm_dict

    def code2desc(self, code):
        icd9cm_dict = self.get_dict()
        ans = icd9cm_dict.get(code, None)
        if not ans:
            print('No such ICD9CM code in 2015!')
        return ans


def read_icd9dict(file='data/CMS32_DESC_LONG_DX.txt', widths=[5,10000]):
    icd9dict = ICD9Desc(file=file, widths=widths)
    return icd9dict.get_dict()