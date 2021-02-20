Search.setIndex({docnames:["index","logging","math","models","planning","replay_buffer","util"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,nbsphinx:3,sphinx:56},filenames:["index.rst","logging.rst","math.rst","models.rst","planning.rst","replay_buffer.rst","util.rst"],objects:{"mbrl.logger":{Logger:[1,1,1,""]},"mbrl.logger.Logger":{log_data:[1,2,1,""],register_group:[1,2,1,""]},"mbrl.math":{Normalizer:[2,1,1,""],Stats:[2,1,1,""],gaussian_nll:[2,3,1,""],propagate:[2,3,1,""],propagate_expectation:[2,3,1,""],propagate_fixed_model:[2,3,1,""],propagate_from_indices:[2,3,1,""],propagate_random_model:[2,3,1,""],truncated_linear:[2,3,1,""],truncated_normal_:[2,3,1,""]},"mbrl.math.Normalizer":{denormalize:[2,2,1,""],load:[2,2,1,""],normalize:[2,2,1,""],save:[2,2,1,""],update_stats:[2,2,1,""]},"mbrl.models":{BasicEnsemble:[3,1,1,""],DynamicsModelTrainer:[3,1,1,""],DynamicsModelWrapper:[3,1,1,""],GaussianMLP:[3,1,1,""],Model:[3,1,1,""],ModelEnv:[3,1,1,""]},"mbrl.models.BasicEnsemble":{eval_score:[3,2,1,""],forward:[3,2,1,""],is_deterministic:[3,2,1,""],load:[3,2,1,""],loss:[3,2,1,""],sample_propagation_indices:[3,2,1,""],save:[3,2,1,""],update:[3,2,1,""]},"mbrl.models.DynamicsModelTrainer":{evaluate:[3,2,1,""],maybe_save_best_weights:[3,2,1,""],train:[3,2,1,""]},"mbrl.models.DynamicsModelWrapper":{eval_score_from_simple_batch:[3,2,1,""],get_output_and_targets_from_simple_batch:[3,2,1,""],predict:[3,2,1,""],update_from_bootstrap_batch:[3,2,1,""],update_from_simple_batch:[3,2,1,""],update_normalizer:[3,2,1,""]},"mbrl.models.GaussianMLP":{eval_score:[3,2,1,""],forward:[3,2,1,""],is_deterministic:[3,2,1,""],load:[3,2,1,""],loss:[3,2,1,""],sample_propagation_indices:[3,2,1,""],save:[3,2,1,""]},"mbrl.models.Model":{eval_score:[3,2,1,""],forward:[3,2,1,""],is_deterministic:[3,2,1,""],load:[3,2,1,""],loss:[3,2,1,""],sample_propagation_indices:[3,2,1,""],save:[3,2,1,""],update:[3,2,1,""]},"mbrl.models.ModelEnv":{evaluate_action_sequences:[3,2,1,""],reset:[3,2,1,""],step:[3,2,1,""]},"mbrl.planning":{Agent:[4,1,1,""],CEMOptimizer:[4,1,1,""],RandomAgent:[4,1,1,""],SACAgent:[4,1,1,""],TrajectoryOptimizer:[4,1,1,""],TrajectoryOptimizerAgent:[4,1,1,""],complete_agent_cfg:[4,3,1,""],create_trajectory_optim_agent_for_model:[4,3,1,""],load_agent:[4,3,1,""]},"mbrl.planning.Agent":{act:[4,2,1,""],plan:[4,2,1,""],reset:[4,2,1,""]},"mbrl.planning.CEMOptimizer":{optimize:[4,2,1,""]},"mbrl.planning.RandomAgent":{act:[4,2,1,""]},"mbrl.planning.SACAgent":{act:[4,2,1,""]},"mbrl.planning.TrajectoryOptimizer":{optimize:[4,2,1,""],reset:[4,2,1,""]},"mbrl.planning.TrajectoryOptimizerAgent":{act:[4,2,1,""],plan:[4,2,1,""],reset:[4,2,1,""],set_trajectory_eval_fn:[4,2,1,""]},"mbrl.replay_buffer":{BootstrapReplayBuffer:[5,1,1,""],IterableReplayBuffer:[5,1,1,""],SimpleReplayBuffer:[5,1,1,""]},"mbrl.replay_buffer.BootstrapReplayBuffer":{add:[5,2,1,""],is_train_compatible_with_ensemble:[5,2,1,""],sample:[5,2,1,""],toggle_bootstrap:[5,2,1,""]},"mbrl.replay_buffer.IterableReplayBuffer":{load:[5,2,1,""]},"mbrl.replay_buffer.SimpleReplayBuffer":{add:[5,2,1,""],is_train_compatible_with_ensemble:[5,2,1,""],load:[5,2,1,""],sample:[5,2,1,""],save:[5,2,1,""]},"mbrl.util":{common:[6,0,0,"-"],mujoco:[6,0,0,"-"]},"mbrl.util.common":{create_dynamics_model:[6,3,1,""],create_replay_buffers:[6,3,1,""],load_hydra_cfg:[6,3,1,""],populate_buffers_with_agent_trajectories:[6,3,1,""],rollout_model_env:[6,3,1,""],save_buffers:[6,3,1,""],step_env_and_populate_dataset:[6,3,1,""],train_model_and_save_model_and_data:[6,3,1,""]},"mbrl.util.mujoco":{freeze_mujoco_env:[6,1,1,""],get_current_state:[6,3,1,""],make_env:[6,3,1,""],make_env_from_str:[6,3,1,""],rollout_mujoco_env:[6,3,1,""],set_env_state:[6,3,1,""]},mbrl:{logger:[1,0,0,"-"],math:[2,0,0,"-"],models:[3,0,0,"-"],planning:[4,0,0,"-"],replay_buffer:[5,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0001":3,"001":3,"06389":4,"12114":[2,3],"1805":[2,3],"1999":4,"200":3,"2008":4,"2018":[2,3],"abstract":[3,4],"boolean":[3,5],"case":[3,6],"class":[1,2,3,4,5,6],"default":[1,2,3,4,5,6],"final":[0,3],"float":[1,2,3,4,5,6],"function":[2,3,4,6],"import":[],"int":[1,2,3,4,5,6],"new":[3,4,5,6],"return":[2,3,4,5,6],"true":[1,2,3,4,5,6],"try":6,"var":2,Doing:3,For:[3,4,5,6],One:3,That:[3,5],The:[1,2,3,4,5,6],Then:3,Using:4,_arg:4,_gener:[5,6],_kwarg:4,_rng:3,_target_:6,_train:6,_val:6,a_t:3,abc:3,about:4,abov:[3,5],access:3,accord:[2,3],accordingli:[2,3],accumul:[3,4],across:3,act:[4,5,6],act_shap:6,act_t:3,action:[3,4,5,6],action_dim:4,action_lb:4,action_rang:4,action_sequ:3,action_shap:5,action_spac:[4,6],action_t:3,action_typ:5,action_ub:4,activ:3,actor:4,actual:3,adam:3,add:5,added:6,after:[3,4,6],again:[3,4],agent:[4,6],agent_cfg:4,agent_kwarg:6,agent_path:4,agent_typ:4,aggreg:3,algorithm:[2,4,6],algorithms_for_calculating_vari:2,all:[2,3,4,5],allow:[3,6],alpha:4,alreadi:1,also:[0,3,4],alwai:6,ani:[3,4,6],anoth:4,ant:6,ant_truncated_ob:6,append:5,appendix:3,appli:4,appropri:[0,4],approxim:3,arg:[3,6],argument:[3,5,6],arrai:[3,4,5,6],arxiv:[2,3,4],assign:[1,3],assum:[3,6],attempt:6,attribut:6,automat:[4,6],averag:[1,3,4],backpropag:3,backward:3,base:[0,1,2,3,4,5,6],base_model:3,basic:3,basicensembl:3,batch:[2,3,4,5,6],batch_siz:[3,5],been:[1,3,6],befor:[3,6],behav:2,behavior:4,being:3,below:3,best:[3,4],best_val_scor:3,best_x:4,between:[1,3,4],beyond:2,blob:4,bool:[1,2,3,4,5,6],bootstrap:[3,5,6],bootstrap_batch:3,bootstrapreplaybuff:[3,5,6],both:6,bound:[3,4],buffer:[0,3,6],build:3,cach:4,call:[1,3,4,6],callabl:[3,4,6],callback:[3,4,6],can:[1,2,3,4,5,6],capabl:4,capac:5,care:4,cartpol:6,cartpole_continu:6,cem:4,cem_pet:4,cemoptim:4,cfg:6,chang:4,check:[4,5],checkpoint:4,child:4,choos:2,chosen:[2,3],chua:[2,3,6],clip:2,clone:0,code:4,collect:[1,6],color:1,com:[0,4],combinatori:4,common:[3,6],complet:[4,5],complete_agent_cfg:4,comput:[3,4],concern:[3,4],config:[4,6],configur:[0,3,4,6],consid:3,consider:3,consist:[3,4],consol:[1,4],constant:2,construct:[3,4],constructor:[3,4,5],contain:[1,3,4,6],content:[0,5],context:6,continu:[4,6],contribut:0,control:6,conveni:[4,6],convert:3,copi:6,core:[3,4,6],correspond:[2,3,4,5,6],could:3,count:2,cours:4,creat:[1,3,4,5,6],create_dynamics_model:6,create_replay_buff:6,create_trajectory_optim_agent_for_model:4,critic:4,cross:4,csv:1,cuda:0,current:[3,4],data:[1,2,3,5,6],dataset:[3,6],dataset_s:6,dataset_train:[3,6],dataset_v:[3,6],davidson:4,deactiv:3,debug:3,decai:3,dedic:1,deep:6,defin:4,delta:3,denorm:2,depend:3,deriv:3,describ:[2,3,4],descript:[4,6],desir:[2,3,4],determin:6,determinist:3,dev:0,develop:0,deviat:[2,4],devic:[2,3,4],dict:[3,4,6],dictconf:6,dictconfig:[3,4,6],dictionari:[1,3,4],differ:3,dim:2,dimens:[2,3,4],directli:3,directori:[1,4,6],distribut:[2,3],divers:3,dm_control:6,dmc2gym:6,dmcontrol:6,dmcontrol___:6,dmcontrol___cheetah:6,do_something_with_batch:5,document:6,doe:3,doesn:3,doing:5,domain:6,don:3,done:[3,5,6],dump:1,dump_frequ:1,dure:3,dynam:[3,6],dynamics_model:[3,6],dynamicsmodeltrain:[3,6],dynamicsmodelwrapp:[3,6],each:[1,2,3,4,5,6],easi:4,effici:3,either:[3,6],elaps:6,element:[3,4],elit:4,elite_ratio:4,els:6,enable_back_compat:1,end:[3,5],ensembl:[2,3,5,6],ensemble_s:[3,5,6],ensur:3,enter:4,entri:[1,4],entropi:4,enumer:3,env:[3,4,6],env_nam:6,env_stat:6,environ:[0,3,4,6],episod:[3,5,6],epistem:3,epoch:3,equal:[3,6],equival:[2,3,4],error:3,eval_scor:3,eval_score_from_simple_batch:3,evalu:[3,4],evaluate_action_sequ:[3,4],everi:[3,4],everytim:5,exactli:3,exampl:[3,6],except:[1,3],execut:6,exit:6,expect:[2,3,6],explain:[3,5],explan:6,extens:5,facil:[],facilit:0,fairintern:0,fals:[1,2,3,4,5,6],far:[3,6],fashion:5,featur:1,field:6,fifo:5,file:[1,4,5,6],fill:4,first:[3,4,6],fixed_model:[2,3],flag:3,float32:5,follow:[2,3,4,6],format:1,forward:3,found:4,freez:6,freeze_mujoco_env:6,frequenc:4,from:[2,3,4,5,6],full:[4,5],func:6,futur:4,gaussian:[2,3],gaussian_nl:2,gaussianmlp:3,gener:[0,3,4,5],get:[3,6],get_current_st:6,get_output_and_targets_from_simple_batch:3,get_stat:6,git:0,github:[0,4],give:3,given:[1,2,3,4,5,6],goal:4,good:4,gradient:3,graph:3,group:1,group_nam:1,gym:[3,4,6],gym___:6,gym___halfcheetah:6,had:6,halfcheetah:6,handl:4,happen:2,has:[0,3],have:[1,3,4,6],header:1,henc:3,hid_siz:3,hidden:[3,4],high:4,higher:3,histori:3,horizon:[3,4],how:[1,3,6],http:[0,2,3,4],humanoid:6,humanoid_truncated_ob:6,hydra:[3,4,6],ident:3,ignor:3,implement:[2,3,4,5,6],importantli:3,improv:3,in_siz:[2,3,6],includ:3,increase_val_set:6,independ:3,index:[0,3,4],indic:[2,3,5,6],info:6,inform:[3,4,6],initi:[3,4,6],initial_mu:4,initial_ob:6,initial_obs_batch:3,initial_st:3,input:[2,3,4,6],instanc:[3,6],instanti:[3,4,6],instead:3,integ:3,intend:3,interfac:3,intern:[4,6],interpret:4,is_determinist:3,is_ensembl:3,is_train_compatible_with_ensembl:5,issu:4,iter:[3,4,5],iterablereplaybuff:[3,5,6],its:[1,3,4,6],janner:6,just:[2,3],keep:[1,2,3,4],keep_last_solut:4,kei:4,kept:[2,3,4],keyword:[1,6],kwarg:[3,6],label:4,last:[1,3,4],latest:6,layer:3,learn:[0,3,6],learned_reward:[3,6],len:[3,6],length:[4,6],level:3,librari:[0,1,4],light:1,like:3,likelihood:[2,3],limit:3,linear:2,list:[1,3,5,6],load:[2,3,4,5,6],load_ag:4,load_dir:6,load_hydra_cfg:6,log:[0,2,3,4],log_data:1,log_dir:1,log_format:1,logger:[1,3],logic:3,logvar:3,look:3,lookahead:6,loop:[3,5],loss:3,low:4,lower:4,lower_bound:4,luisenp:4,made:3,make:[0,4,6],make_env:6,make_env_from_str:6,mani:[3,6],manipul:[3,6],manual:[4,6],map:1,master:4,match:3,math:[0,3],max:[2,4],max_i:2,max_log_var:3,max_logvar:3,max_x:2,maxim:4,maximum:[3,4,5,6],maybe_save_best_weight:3,mbpo:6,mbrl:[1,2,3,4,5,6],mean:[2,3,4],measur:1,member:[3,6],member_cfg:3,meta:6,metadata:3,meth:[],method:[1,2,3,4,6],methodolog:4,might:[4,6],min:[2,4],min_i:2,min_log_var:3,min_logvar:3,min_x:2,mind:6,mlp:3,mode:3,model:[0,2,4,5,6],model_arg_1:6,model_arg_n:6,model_batch_s:6,model_dir:6,model_env:[4,6],model_in:3,model_train:6,modelenv:[3,4,6],modif:1,modifi:2,modul:0,moment:3,momentum:4,more:6,moreov:3,mostli:3,mse:3,mse_loss:3,mujoco:6,multi:3,multipl:3,mus:4,must:[1,2,3,4],name:[1,5,6],ndarrai:[2,3,4,5,6],need:[2,3,4,6],neg:[2,3],neurip:[2,3],next:[3,4,5,6],next_ob:[3,5,6],next_observ:3,nll:3,no_delta_list:[3,6],no_grad:3,non:3,none:[2,3,4,5,6],normal:[2,3,6],note:[2,3,4],now:4,npz:[5,6],num_epoch:[3,6],num_epochs_train_model:6,num_iter:4,num_lay:3,num_memb:[3,5],num_particl:[3,4],num_sampl:6,num_trial:6,number:[2,3,4,5,6],numer:3,numpi:[2,3,4,5,6],o1_expect:6,o_t:3,obj_fun:4,object:[1,2,3,4,5,6],obs:[3,4,5,6],obs_dim:4,obs_process_fn:[3,6],obs_shap:[5,6],obs_t:3,obs_typ:5,observ:[3,4,5,6],observation_spac:4,obtain:6,occur:4,often:1,omegaconf:[3,4,6],onc:3,one:[1,3,5,6],onli:[2,3,4,5,6],onlin:2,oper:[3,4],oppos:4,optim:[3,4],optim_lr:3,optimizer_cfg:4,option:[1,2,3,4,5,6],order:5,org:[2,3,4],origin:[3,6],otherwis:[1,3,4,6],out:[3,6],out_siz:[3,6],output:[1,2,3,6],outsid:4,over:[2,3,4,5,6],overrid:6,overridden:[3,4,6],overwritten:5,own:[1,3],page:0,paper:[2,3,6],parallel:3,paramet:[1,2,3,4,5,6],particl:[3,4],particular:3,pass:[1,3,4,6],path:[1,2,3,4,5,6],pathlib:[1,2,4,6],patienc:[3,6],pdf:[2,3,4],per:[3,5],perceptron:3,perform:[3,4],permut:3,pet:[2,3,6],pets_halfcheetah:6,physic:6,pickl:6,pip:0,place:2,plan:[0,6],planning_horizon:4,plot:4,pointer:2,polici:4,popul:[4,6],populate_buffers_with_agent_trajectori:6,population_s:4,posit:[3,6],pre:[3,6],preced:6,pred_logvar:2,pred_mean:2,pred_obs_:3,pred_rewards_:3,predict:[2,3],predicted_tensor:2,prefix:6,present:6,previou:4,print:4,probabilist:3,probabl:[3,4,6],problem:4,process:[3,4,6],produc:3,product:6,propag:[2,3,4],propagate_expect:2,propagate_fixed_model:2,propagate_from_indic:2,propagate_random_model:2,propagation_indic:[2,3],propagation_method:[2,3,4],proport:[4,6],provid:[3,4,5,6],pth:[4,6],purpos:[3,4],pytest:0,python:[0,6],pytorch:0,pytorch_sac:[1,4],random:[2,3,5,6],random_choic:2,random_model:[2,3,4],randomag:4,randomli:3,rang:4,rare:2,rate:3,rather:5,reach:5,read:6,receiv:[3,4],recov:3,reduc:3,reduct:3,regist:1,register_group:1,regularli:6,reinforc:0,rel:3,relu:3,remov:[1,3],repeat:4,replac:[4,5],replai:[0,3,6],replan_freq:4,replay_buff:[3,5,6],replay_buffer_train:6,replay_buffer_v:6,replic:3,repositori:0,repres:[1,2,3,4],requir:[4,5],research:0,reset:[3,4,6],resid:2,respect:[2,3,4,5,6],result:[3,6],results_dir:[2,6],return_as_np:3,reward:[3,4,5,6],reward_fn:[3,6],rng:[3,5,6],roll:[3,6],rollout:3,rollout_env:[],rollout_model_env:6,rollout_mujoco_env:6,round:4,rubinstein:4,run:[0,2,3,4,6],s_online_algorithm:2,sac:4,sac_ag:4,sacag:4,same:[2,3,6],sampl:[2,3,4,5,6],sample_propagation_indic:3,save:[1,2,3,4,5,6],save_buff:6,save_dir:2,score:3,search:0,second:[3,4,6],section:4,see:[3,6],seed:3,select:6,self:[3,4],sent:3,separ:1,sequenc:[3,4,6],set:[3,4,6],set_env_st:6,set_trajectory_eval_fn:4,shape:[2,3,4,5,6],shift:4,shortcut:1,should:[1,3,4,6],shuffl:5,shuffle_each_epoch:5,sign:3,signal:4,silu:3,simplereplaybuff:[5,6],simpli:[3,5],simul:6,sinc:[1,3],singl:[3,5],size:[2,3,4,5,6],slice:3,small:3,smaller:3,soft:4,solut:4,some:[1,3,6],space:4,specif:5,specifi:[5,6],sqrt:2,squar:3,stack:3,standard:[2,4,5],start:6,stat:2,state:[3,4,6],statist:[2,3],std:2,step:[3,4,6],step_count:6,step_env_and_populate_dataset:6,step_the_env_a_bunch_of_tim:6,steps_to_collect:6,stop:3,store:[2,3,5],str:[1,2,3,4,5,6],string:[1,6],structur:6,style:3,subclass:[3,5],subsequ:4,suit:6,support:[3,4],sure:0,system:0,tabular:1,take:[2,4,6],target:[2,3],target_is_delta:[3,6],task:6,tensor:[1,2,3,4,6],term:[3,4],term_fn:6,termin:6,termination_fn:[3,6],test:0,text:[],th3:4,than:[3,5],thei:3,them:3,thi:[1,2,3,4,5,6],threshold:3,through:3,thrown:[1,3],time:[1,3,4,5],time_limit:6,timelimit:6,toggl:5,toggle_bootstrap:5,tool:0,torch:[1,2,3,4,6],total:[3,4],track:[1,3],train:[3,5,6],train_buff:6,train_dataset:6,train_is_bootstrap:6,train_model_and_save_model_and_data:6,trainer:[3,5,6],trajectori:[3,4],trajectory_eval_fn:4,trajectoryoptim:4,trajectoryoptimizerag:4,transit:[3,5,6],tri:6,trial:6,trial_length:6,truncat:2,truncated_linear:2,truncated_normal_:2,ts1:[2,3],tsinf:[2,3],tupl:[1,2,3,4,5,6],turn:4,two:[2,3,6],type:[1,2,3,4,5,6],uncertainti:[3,4],under:6,underli:[3,4],unexpect:4,union:[1,2,3,4,5,6],unless:4,until:3,updat:[2,3],update_from_bootstrap_batch:3,update_from_simple_batch:3,update_norm:3,update_stat:2,upon:6,upper:4,upper_bound:4,usag:6,use:[1,2,3,4,5,6],use_silu:3,use_train_set:3,used:[1,2,3,4,5,6],user:[3,4,6],uses:[2,4],using:[1,2,3,4,6],util:[0,3,4],val:[2,6],val_buff:6,val_dataset:6,val_ratio:6,val_scor:3,valid:[2,3,6],validation_ratio:6,valu:[2,3,4,5,6],value_max:4,value_mean:4,value_std:4,variabl:[1,4],variable_nam:1,varianc:[2,3],veloc:6,verbos:4,version:[3,4,6],via:4,wai:3,want:0,weight:[1,3,6],weight_decai:3,welford:2,what:6,when:[1,3,4,5,6],where:[1,2,3,4],whether:[3,5,6],which:[2,3,4,6],whole:3,wiki:2,wikipedia:2,without:[3,4],work:6,work_dir:6,would:6,wrap:[3,4,6],wrapper:[3,4,6],x_shape:4,yaml:[4,6],yellow:1,you:[0,1],your:0,zero_grad:3},titles:["Documentation for mbrl-lib","Logging module","Math utilities module","Models module","Planning module","Replay buffer module","General utilities module"],titleterms:{buffer:5,document:0,gener:6,get:0,indic:0,instal:0,lib:0,log:1,math:2,mbrl:0,model:3,modul:[1,2,3,4,5,6],plan:4,replai:5,start:0,tabl:0,util:[2,6]}})