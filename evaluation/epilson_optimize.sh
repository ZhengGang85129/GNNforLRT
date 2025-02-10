#!/bin/bash
#DBSCAN_config=$(1)
#epsilon_sample_points=("0p05" "0p15" "0p20" "0p21" "0p22" "0p23" "0p24" "0p25" "0p26" "0p27" "0p28" "0p29" "0p30" "0p31" "0p32" "0p33" "0p34" "0p35" "0p45" "0p55" "0p65" "0p75" "0p85")
#epsilon_sample_points=("0p15" "0p20" "0p21" "0p22" "0p23" "0p24" "0p25" "0p26" "0p27" "0p28" "0p29" "0p30" "0p31" "0p32" "0p33" "0p34" "0p35" "0p45" "0p55" "0p65" "0p75" "0p85")
epsilon_sample_points=("0p05" "0p15" "0p25" "0p35" "0p36" "0p37" "0p38" "0p39" "0p40" "0p41" "0p42" "0p43" "0p44" "0p45" "0p55" "0p56" "0p57" "0p58" "0p59" "0p60" "0p61" "0p62" "0p63" "0p64" "0p65" "0p66" "0p67" "0p68" "0p69" "0p70" "0p75" "0p85")
epsilon_sample_points=("0p05" "0p06" "0p07" "0p08" "0p09" "0p10" "0p11" "0p12" "0p13" "0p14" "0p15" "0p16" "0p17" "0p18" "0p19" "0p20" "0p25" "0p30" "0p35" "0p40" "0p45" "0p50" "0p55" "0p60" "0p65" "0p70" "0p75" "0p80" "0p85" "0p90" "0p95")
#epsilon_sample_points=("0p30" "0p31" "0p32" "0p33" "0p34" "0p36" "0p37" "0p38" "0p39")
for epsilon in "${epsilon_sample_points[@]}"; do 
    python3 ./tracks/track_reconstruction_DBSCAN_optimize_search.py tracks/DBSCAN_config/all_PU200.yaml $epsilon > dbscan_optimize_$epsilon.log
done
python3 tracks/epsilon_pur_eff.py
#code ../metrics/epsilon_scan/DBSCAN_eff_pur.pdf

