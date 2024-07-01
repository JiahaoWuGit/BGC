############################defense
#cora 
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2 > logs/cora-defense/cora-gcond-r025-cluster-20240601-prune02.txt &
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=1  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2 > logs/cora-defense/cora-gcond-r05-cluster-20240601-prune02.txt &
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=2  --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/cora-defense/cora-gcond-r1-cluster-20240601-prune02.txt &

# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=3  --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2 > logs/cora-defense/cora-gcond-r025-cluster-20240601-rand_smooth02.txt &
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=4  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2 > logs/cora-defense/cora-gcond-r05-cluster-20240601-rand_smooth02.txt &
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=5  --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/cora-defense/cora-gcond-r1-cluster-20240601-rand_smooth02.txt &

#citeseer
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=6  --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/citeseer-defense/citeseer-gcond-r025-cluster-20240601-prune02.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=7  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/citeseer-defense/citeseer-gcond-r05-cluster-20240601-prune02.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/citeseer-defense/citeseer-gcond-r1-cluster-20240601-prune02.txt &

# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=3  --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/citeseer-defense/citeseer-gcond-r025-cluster-20240601-rand_smooth02.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=4  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/citeseer-defense/citeseer-gcond-r05-cluster-20240601-rand_smooth02.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=5  --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/citeseer-defense/citeseer-gcond-r1-cluster-20240601-rand_smooth02.txt &

# flickr
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.01 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/flickr-defense/flickr-gcond-r001-cluter-20240602-rand_smooth02.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.005 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/flickr-defense/flickr-gcond-r0005-cluter-20240602-rand_smooth02.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.001 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/flickr-defense/flickr-gcond-r0001-cluter-20240602-rand_smooth02.txt &

# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.01 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/flickr-defense/flickr-gcond-r001-cluter-20240602-prune02.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.005 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/flickr-defense/flickr-gcond-r0005-cluter-20240602-prune02.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.001 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/flickr-defense/flickr-gcond-r0001-cluter-20240602-prune02.txt &

#reddit
# # python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/reddit-defense/reddit-gcond-r00005-cluter-20240602-lr_trigger001-lr_feat01-rand_smooth02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.0005 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/reddit-defense/reddit-gcond-r00005-cluter-20240602-lr_trigger001-lr_feat02-rand_smooth02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.001 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/reddit-defense/reddit-gcond-r0001-cluter-20240602-lr_trigger001-lr_feat01-rand_smooth02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.008 --trigger_size 4 --selector 'cluster' --defense_type 'rand_smooth' --prune_rate 0.2  > logs/reddit-defense/reddit-gcond-r0002-cluter-20240602-lr_trigger0008-lr_feat01-rand_smooth02.txt &

# # python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2   > logs/reddit-defense/reddit-gcond-r00005-cluter-20240602-lr_trigger001-lr_feat01-prune02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.0005 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/reddit-defense/reddit-gcond-r00005-cluter-20240602-lr_trigger001-lr_feat02-prune02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.001 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/reddit-defense/reddit-gcond-r0001-cluter-20240602-lr_trigger001-lr_feat01-prune02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.008 --trigger_size 4 --selector 'cluster' --defense_type 'prune' --prune_rate 0.2  > logs/reddit-defense/reddit-gcond-r0002-cluter-20240602-lr_trigger0008-lr_feat01-prune02.txt &
############################defense



############################based clustering select
#cora
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/cora/cora-gcond-r025-cluster-20240606-for-save.txt &
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=1  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/cora/cora-gcond-r05-cluster-20240606-for-save.txt &
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=2  --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/cora/cora-gcond-r1-cluster-20240606-for-save.txt &

#citeseer###########
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=3e-4 --gpu_id=2 --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=750 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r025-cluster-20240609-for-save-lr_feat3e-4.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=4  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r05-cluster-20240606-for-save.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=7e-5 --gpu_id=5 --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=200 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r1-cluster-20240609-for-save-lr_feat7e-5.txt &

# flickr
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.01 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r001-cluter-20240606-for-save.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.007 --lr_adj=0.01  --r=0.005 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save-lr_trigger001-lr_feat00007.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.001 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0001-cluter-20240606-for-save.txt &


#reddit
# #0.0005
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240609-lr_trigger001-lr_feat01-for-save.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.0005 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240609-lr_trigger001-lr_feat02-for-save-seed1.txt &
# # #0.001
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.001 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0001-cluter-20240609-lr_trigger001-lr_feat01-for-save-seed1.txt &
# # #0.002
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.008 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240609-lr_trigger0008-lr_feat01-for-save-seed1.txt &
############################based clustering select




#finetune for citeseer 0.25 & 1.0

# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=3  --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r025-cluster-20240606-for-save.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=3e-4 --gpu_id=1 --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r025-cluster-20240609-for-save-lr_feat3e-4.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=7e-5 --gpu_id=5 --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r025-cluster-20240609-for-save-lr_feat7e-5.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=5e-4 --gpu_id=7 --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r025-cluster-20240609-for-save--lr_feat5e-4.txt &

# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=3e-4 --gpu_id=1 --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r1-cluster-20240609-for-save-lr_feat3e-4.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=7e-5 --gpu_id=5 --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r1-cluster-20240609-for-save-lr_feat7e-5.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=5e-4 --gpu_id=7 --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r1-cluster-20240609-for-save--lr_feat5e-4.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=5  --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r1-cluster-20240606-for-save.txt &




#finetune for flickr r=0.005
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.005 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save-lr_trigger005.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.005 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save-lr_trigger001.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.005 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save-lr_trigger05.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.005 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save-lr_trigger03.txt &

# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.007 --lr_adj=0.01  --r=0.005 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save-lr_trigger005-lr_feat00007.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.007 --lr_adj=0.01  --r=0.005 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save-lr_trigger05-lr_feat00007.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.007 --lr_adj=0.01  --r=0.005 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save-lr_trigger03-lr_feat00007.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.005 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save.txt &



############################the followings are useless parameter tuning
#reddit parameter tuning
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.001 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0001-cluter-20240606-.txt & 
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.002 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240606-.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.0005 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240606-.txt & 

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.001 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0001-cluter-20240606-lr_trigger01-lr_feat005.txt & 
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.002 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240606-lr_trigger01-lr_feat005.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.0005 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240606-lr_trigger01-lr_feat005.txt & 

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.01 --lr_adj=0.1  --r=0.001 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0001-cluter-20240606-lr_trigger01-lr_feat001.txt & 
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.01 --lr_adj=0.1  --r=0.002 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240606-lr_trigger01-lr_feat001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.01 --lr_adj=0.1  --r=0.0005 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240606-lr_trigger01-lr_feat001.txt & 

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.001 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0001-cluter-20240606-lr_trigger01-lr_feat01.txt & 
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240606-lr_trigger01-lr_feat01.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240606-lr_trigger01-lr_feat01.txt & 

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240526-lr_trigger005-lr_feat01.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.01 --lr_adj=0.1  --r=0.002 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240526-lr_trigger005-lr_feat001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.002 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240526-lr_trigger005-lr_feat005.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.001 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0001-cluter-20240526-lr_trigger005-lr_feat01.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.01 --lr_adj=0.1  --r=0.001 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0001-cluter-20240526-lr_trigger005-lr_feat001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.001 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0001-cluter-20240526-lr_trigger005-lr_feat005.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240526-lr_trigger005-lr_feat01.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.01 --lr_adj=0.1  --r=0.0005 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240526-lr_trigger005-lr_feat001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.0005 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240526-lr_trigger005-lr_feat005.txt &

#continue to fine-tune 20240527
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.0005 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger005-lr_feat02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.0005 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger005-lr_feat03.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.0005 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger005-lr_feat04.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.0005 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger005-lr_feat007.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.0005 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger01-lr_feat007.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger01-lr_feat01.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.0005 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger01-lr_feat02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.0005 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger01-lr_feat03.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.0005 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger01-lr_feat04.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.0005 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger001-lr_feat007.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger001-lr_feat01.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.0005 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger001-lr_feat02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.0005 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger001-lr_feat03.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.0005 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger001-lr_feat04.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.0005 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger007-lr_feat007.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger007-lr_feat01.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.0005 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger007-lr_feat02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.0005 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger007-lr_feat03.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.0005 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger007-lr_feat04.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.0005 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger03-lr_feat007.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger03-lr_feat01.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.0005 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger03-lr_feat02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.0005 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger03-lr_feat03.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.0005 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240527-lr_trigger03-lr_feat04.txt &


# #0.002  continue to fine-tune 20240530
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.002 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger001-lr_feat007.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger001-lr_feat01.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.002 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger001-lr_feat02.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.002 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger03-lr_feat03.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.002 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger03-lr_feat04.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.002 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger01-lr_feat007.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger01-lr_feat01.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.002 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger005-lr_feat005.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger005-lr_feat01.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.01 --lr_adj=0.1  --r=0.002 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger005-lr_feat001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger007-lr_feat01.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.002 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger005-lr_feat02.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.002 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger005-lr_feat03.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.002 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger005-lr_feat04.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.002 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger005-lr_feat007.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.002 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger01-lr_feat02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.002 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger01-lr_feat03.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.002 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger01-lr_feat04.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.002 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger001-lr_feat03.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.002 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger001-lr_feat04.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.002 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger007-lr_feat007.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.002 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger03-lr_feat02.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.002 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger007-lr_feat02.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.3 --lr_adj=0.1  --r=0.002 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger007-lr_feat03.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.4 --lr_adj=0.1  --r=0.002 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.07 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger007-lr_feat04.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.07 --lr_adj=0.1  --r=0.002 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger03-lr_feat007.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.3 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240530-lr_trigger03-lr_feat01.txt &



#0.002  continue to fine-tune 20240531
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.001 --lr_adj=0.1  --r=0.002 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger001-lr_feat0001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.0005 --lr_adj=0.1  --r=0.002 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger001-lr_feat00005.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.0001 --lr_adj=0.1  --r=0.002 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger001-lr_feat00001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.005 --lr_adj=0.1  --r=0.002 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger001-lr_feat0005.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=1e-5 --lr_adj=0.1  --r=0.002 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger001-lr_feat1e-5.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=5e-5 --lr_adj=0.1  --r=0.002 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger001-lr_feat5e-5.txt &



# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.001 --lr_adj=0.1  --r=0.002 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger005-lr_feat0001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.0005 --lr_adj=0.1  --r=0.002 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger005-lr_feat00005.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.0001 --lr_adj=0.1  --r=0.002 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger005-lr_feat00001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.005 --lr_adj=0.1  --r=0.002 --gpu_id=4 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger005-lr_feat0005.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=1e-5 --lr_adj=0.1  --r=0.002 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger005-lr_feat1e-5.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=5e-5 --lr_adj=0.1  --r=0.002 --gpu_id=5 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger005-lr_feat5e-5.txt &



# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.001 --lr_adj=0.1  --r=0.002 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.005 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger0005-lr_feat0001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.0005 --lr_adj=0.1  --r=0.002 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.005 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger0005-lr_feat00005.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.0001 --lr_adj=0.1  --r=0.002 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.005 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger0005-lr_feat00001.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.005 --lr_adj=0.1  --r=0.002 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.005 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger0005-lr_feat0005.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=1e-5 --lr_adj=0.1  --r=0.002 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.005 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger0005-lr_feat1e-5.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=5e-5 --lr_adj=0.1  --r=0.002 --gpu_id=6 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.005 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger0005-lr_feat5e-5.txt &

# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.002 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger05-lr_feat005.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.002 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger01-lr_feat005.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.05 --lr_adj=0.1  --r=0.002 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --lr_trigger 0.05 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240531-lr_trigger005-lr_feat005.txt &
