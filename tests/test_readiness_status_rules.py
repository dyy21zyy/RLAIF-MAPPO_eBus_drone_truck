from experiments.run_readiness_pilot import classify
def test_formal_experiments_not_from_diagnostic():
 gates={k:True for k in ['event','passenger','power','reward','metrics','mappo','checkpoint']}; assert classify(gates,False)!='FORMAL_EXPERIMENTS_COMPLETED'; assert classify(gates,True,formal_done=True)=='FORMAL_EXPERIMENTS_COMPLETED'
