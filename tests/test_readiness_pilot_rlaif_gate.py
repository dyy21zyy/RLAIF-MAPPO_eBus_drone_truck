from experiments.run_readiness_pilot import classify
def test_rlaif_status_rules():
 gates={'event':True,'passenger':True,'power':True,'reward':True,'metrics':True,'mappo':True,'checkpoint':True}; assert classify(gates,False)=='ENV_MAPPO_READY_RLAIF_BLOCKED'; assert classify(gates,True)=='READY_FOR_FORMAL_TRAINING'; gates['event']=False; assert classify(gates,True)=='NOT_READY'
