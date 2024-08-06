import unittest
from utils import update_cols, get_dfs, create_idxs, scale_df, load_csvs, features_extraction
import pandas as pd


def get_missing_steps(df):
    max_step = df['step'].max()
    unique_steps = set(df['step'].unique())
    needed = set(range(max_step + 1))
    missing = needed - unique_steps
    return missing

class TestStringMethods(unittest.TestCase):


    def test_steps_in_load_csvs(self):
        ues, cells = load_csvs()
        # ue_steps = [ue['step'].unique() for ue in ues]
        # cell_steps = [cell['step'].unique() for cell in cells]

        for ue in ues:
            print(get_missing_steps(ue))


    def test_dataframes(self):
        ues, cells = load_csvs()

        for ue in ues:
            self.assertEqual({50}, set(ue.groupby('step')['ue-id'].count()), msg=f"some steps doesnt have 50 ues rows")
            # test if there are 50 unique ues every step
            for s in ue['step'].unique():
                self.assertEqual(50, len(ue[ue['step'] == s]['ue-id'].unique()),
                                 msg=f"some steps doesnt have 50 unique ues fail at step {s}")


if __name__ == '__main__':
    unittest.main()
