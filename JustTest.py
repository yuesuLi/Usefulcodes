import os
import json
import argparse
import pandas as pd

def test(aa):
    aa.remove(1)

def main():
    df_results = pd.DataFrame(columns=['results', 'groupname'])
    df_result = pd.DataFrame(columns=['results', 'groupname'])
    result_str = 'MOTA={}, MOTP={}'.format(round(3.1415926, 4), round(8.54613, 4))
    df_result['results'] = [result_str]
    df_result['groupname'] = ['Hello World!']
    df_results = pd.concat((df_results, df_result))
    a = [1,2,3]
    test(a)
    print()


if __name__ == '__main__':
    main()