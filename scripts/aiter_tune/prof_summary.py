import sys, pandas as pd
d=pd.read_csv(sys.argv[1])
H1="flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui"; H2="flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2"
for t,g in d.groupby("token"):
    s1=g[g.stage=="stage1"]; s2=g[g.stage=="stage2"]
    h1=s1[(s1.kernelName==H1)&(s1.block_m==32)].us.min(); h2=s2[(s2.kernelName==H2)&(s2.block_m==32)].us.min()
    print(t, "heur", h1, h2, round(h1+h2,1), "| best s1", s1.us.min(), "best s2", s2.us.min(), "n", len(g))
