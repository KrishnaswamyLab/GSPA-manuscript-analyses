for model in GFMMD Eigenscore GAE_att_Ggene Signals GSPA GSPA_QR MAGIC DiffusionEMD Node2Vec_Ggene GAE_noatt_Ggene SIMBA siVAE; do
    echo ${model}
    for dataset in linear 2_branches 3_branches; do
        python evaluate_localization.py ${model} ${dataset}
    done
done