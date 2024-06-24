for model in Eigenscore GFMMD Signals DiffusionEMD GSPA GSPA_QR MAGIC Node2Vec_Ggene GAE_noatt_Ggene GAE_att_Ggene SIMBA siVAE; do
    echo ${model}
    for dataset in linear 2_branches 3_branches; do
        python evaluate_coexpression.py ${model} ${dataset}
    done
done
