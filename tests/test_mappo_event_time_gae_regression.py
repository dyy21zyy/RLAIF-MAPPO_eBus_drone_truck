from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition

def add(b, event, eid, t):
    agent={0:"assignment",1:"truck",2:"bus",3:"bus",4:"station"}[eid]
    b.append(AsyncTransition(agent,[0.],[0.],0,[True],[[0.]],("x",),0.,1.,1.,False,[0.],event,t,eid,total_reward=1.))

def test_event_time_discount_alignment():
    b=AsyncMAPPOBuffer(); add(b,"BUS_TERMINAL_DEPARTURE",2,0.); add(b,"BUS_STATION_ARRIVAL",3,10.)
    returns, adv=b.compute_returns_and_advantages(gamma=0.5, gae_lambda=1.0, reference_time_unit=10.0, per_agent_normalize=False)
    assert b.transitions[0].event_type_id==2 and b.transitions[1].event_type_id==3
    assert returns[0] < 2.0
