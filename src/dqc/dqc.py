import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import datetime as dt
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQCPipeline():

    
    def validation_report(self, data:Dict[str, pd.DataFrame]):
        
        validation_report = {}
        validation_report['missing_values'] = self._check_missing_values(data)
        validation_report['data_types'] = self._check_data_types(data)
        validation_report['uniques'] = self._check_uniques(data)
        validation_report['duplicates'] = self._check_duplicates(data)
        validation_report['consistency'] = self._check_consistency(data)

        return validation_report
    
    def statistics_report(self, data:Dict[str, pd.DataFrame]):
        

        statistics_report = {}
        statistics_report['statistics'] = self._check_statistics(data)
        statistics_report['outliers'] = self._check_outliers(data)


        return statistics_report


    def render_report(self, validation_report, statistics_report):
        report = {
            'validation_report': validation_report,
            'statistics_report': statistics_report
        }
        return report
    

    def _check_missing_values(self, data:Dict[str, pd.DataFrame]):
        report = []

        for table_name, df in data.items():
            missings = df.isna().sum() / len(df)
            missings_df = missings.reset_index()
            missings_df.columns = ["column_name", "missings_percentage"]
            missings_df["table_name"] = table_name
            report.append(missings_df)

        result = pd.concat(report, ignore_index=True)
        return result.set_index(["table_name", "column_name"])
    
    def _check_uniques(self, data:Dict[str, pd.DataFrame]):

        report = []

        for table_name, df in data.items():
            uniques = df.nunique()
            uniques_df = uniques.reset_index()
            uniques_df.columns = ["column_name", "unique_values"]
            uniques_df["table_name"] = table_name
            report.append(uniques_df)

        result = pd.concat(report, ignore_index=True)
        return result.set_index(["table_name", "column_name"])
    

    def _check_data_types(self, data:Dict[str, pd.DataFrame]):
        report = []

        for t_name, df in data.items():
            current_dtypes = df.dtypes.astype(str).values
            infer_dtypes = df.infer_objects().dtypes.astype(str).values

            is_mixed = current_dtypes != infer_dtypes
            report.append(
                    pd.DataFrame(
                        data=[current_dtypes, infer_dtypes, is_mixed],
                        index=["current_dtype", "inferred_dtype", "is_mixed"],
                        columns=pd.MultiIndex.from_tuples(
                            [(t_name, col) for col in df.columns],
                            names=["table_name", "column_name"]
                        )
                    )
                )

        return (
            pd.concat(report, axis=1)
            .transpose()
            .reset_index()
            .sort_values("table_name")
            .set_index(["table_name", "column_name"])
        )

    def _check_duplicates(self, data:Dict[str, pd.DataFrame]):

        duplicates = {}
        for table, df in data.items():
            duplicates[table] = df.duplicated().sum()           
        return pd.DataFrame.from_dict(data=duplicates, orient="index", columns=["duplicated_rows"])
            

    def _check_consistency(self, data: Dict[str, pd.DataFrame]):

        results = []
        table_names = list(data.keys())

        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                t1_name, t2_name = table_names[i], table_names[j]
                t1, t2 = data[t1_name], data[t2_name]
                
                common_cols = set(t1.columns).intersection(set(t2.columns))
                for col in common_cols:
                    left_ids = set(t1[col].dropna().unique())
                    right_ids = set(t2[col].dropna().unique())

                    intersect = left_ids & right_ids
                    left_only = left_ids - right_ids
                    right_only = right_ids - left_ids

                    results.append({
                        "relation": f"{t1_name}, {t2_name} ON {col}",
                        "left_only": len(left_only),
                        "intersect": len(intersect),
                        "right_only": len(right_only)
                    })

        df_results = pd.DataFrame(results)
        return df_results
    

    def _check_outliers(self, data: Dict[str, pd.DataFrame]):
        report = []

        for table_name, df in data.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                logger.info(f'No numeric columns in table "{table_name}" - {dt.datetime.now()}')
                continue

            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_percentage = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]) / len(df) * 100

                outlier_mapping = {
                    'table': table_name,
                    'column': col,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outliers_percentage': outlier_percentage,
                    'min_val': df[col].min(),
                    'max_val': df[col].max(),
                    '1%_percentile': df[col].quantile(0.01),
                    '99%_percentile': df[col].quantile(0.99)
                }
                report.append(outlier_mapping)

        if not report:
            logger.info(f'No numeric data found in any table - {dt.datetime.now()}')
            return pd.DataFrame()

        result = pd.DataFrame(report)
        return result.set_index(["table", "column"])
    
    def _check_statistics(self, data: Dict[str, pd.DataFrame], percentiles: List[float] = [0.01, 0.25, 0.75, 0.99]):

        report = []

        for table_name, df in data.items():
            desc = df.describe(percentiles=percentiles).transpose()
            desc["table_name"] = table_name
            desc["column_name"] = desc.index
            report.append(desc)

        result = pd.concat(report, ignore_index=True)
        return result.set_index(["table_name", "column_name"])