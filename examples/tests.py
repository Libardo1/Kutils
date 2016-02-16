import kutils as kutils
import numpy as np
import pandas as pd 

def test_with_pandas():

	df = pd.read_csv("train.csv")

	# kutils.plotPCA(df.drop(["Id", "Response", "Product_Info_2"],1), df["Response"])
	# kutils.plotCorr(df.drop(["Id", "Response", "Product_Info_2"],1), df["Response"], fontsize = 6)
	# kutils.plotCorr(df.drop(["Id", "Response", "Product_Info_2"],1).values, fontsize = 6)
	# kutils.plot1D(df[["Wt", "BMI", "Ht"]])
	# kutils.plot1D(df[["Wt", "BMI", "Ht"]].values, list_feat = ["Wt", "BMI", "Ht"])
	# kutils.plot1Dbyclass(df[["Wt", "BMI", "Ht"]], df["Response"], single_out=[1,2])
	# kutils.plot1Dbyclass(df[["Wt", "BMI", "Ht"]].values, df["Response"].values)
	# kutils.plotViolinbyclass(df[["BMI", "Ht", "Wt"]], df["Response"])
	# kutils.plotViolinbyclass(df[["BMI", "Ht", "Wt"]].values, df["Response"].values,list_feat = ["A", "B", "C"])

	# kutils.plotFeatImpKBest(df[["BMI", "Ht", "Wt"]], df["Response"])
	# kutils.plotFeatImpXGB(df.drop(["Id", "Response", "Product_Info_2"],1), df["Response"], yticklabelsize=6)

	# kutils.plot2D(df, feat1 = "BMI", feat2 = "Wt")
	# kutils.plot2Dbyclass(df[["BMI", "Wt"]].values,df["Response"].values, feat1 = "BMI", feat2 = "Wt", single_out=[1,2])
	kutils.get_summary_statistics(df)

if __name__ == '__main__':
	
	test_with_pandas()