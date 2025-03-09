import os

wd = "/home/kasper/data/asthma/No_hos/"

"""
cmd1 = "plink --bfile " + wd + "No_hos_clump --genome --out " + wd + "No_hos_clump"
os.system(cmd1)

cmd2 = "plink --bfile " + wd + "No_hos_clump "
cmd3 = "--read-genome " + wd + "No_hos_clump.genome "
cmd4 = "--cluster --mds-plot 10 --out " + wd + "No_hos_clump"
os.system(cmd2+cmd3+cmd4)


cmd5 = "awk '{print$1, $2, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13}' "
cmd6 = wd + "No_hos_clump.mds > mds.txt"
os.system(cmd5+cmd6)
"""

cmd7 = "plink --bfile " + wd + "No_hos_clump " 
cmd8 = "--covar "+wd+"mds.txt --logistic --sex --hide-covar --adjust --out "
cmd9 = wd + "No_hos_clump"
os.system(cmd7+cmd8+cmd9)

cmd10 = "awk '!/'NA'/' "
cmd11 = wd + "No_hos_clump.assoc.logistic > "
cmd12 = wd + "No_hos_clump.assoc_2.logistic"
os.system(cmd10+cmd11+cmd12)

cmd13 = "awk '!/'NA'/' "
cmd14 = wd + "No_hos_clump.assoc.logistic.adjusted > "
cmd15 = wd + "No_hos_clump.assoc_2.logistic.adjusted"
os.system(cmd13+cmd14+cmd15)